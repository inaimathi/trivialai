"""
OpenRouter adapter: text + vision via Chat Completions, images via /images.

No network: see _http.Recorder.
"""

import base64
import unittest

from _http import Recorder
from _png import make_png

import src.trivialai as trivialai
from src.trivialai import openrouter as openrouter_mod
from src.trivialai.chatgpt import ChatGPT
from src.trivialai.image import Picture
from src.trivialai.openrouter import OpenRouter

RED = make_png(rgba=(255, 0, 0, 255))
GREEN = make_png(rgba=(0, 255, 0, 255))
BLUE = make_png(rgba=(0, 0, 255, 255))

BASE = "https://openrouter.ai/api/v1"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _decode_ref(ref: dict) -> bytes:
    assert ref["type"] == "image_url", ref
    return base64.b64decode(ref["image_url"]["url"].split(",", 1)[1])


class TestOpenRouterText(unittest.TestCase):
    def _client(self):
        return OpenRouter(model="anthropic/claude-test", api_key="or-test")

    def test_is_a_chatgpt_client_pointed_at_openrouter(self):
        c = self._client()
        self.assertIsInstance(c, ChatGPT)
        self.assertEqual(c.base_url, BASE)
        self.assertEqual(OpenRouter(api_key="k", base_url="").base_url, BASE)

    def test_generate_with_images(self):
        rec = Recorder(json_body={"choices": [{"message": {"content": "same knight"}}]})
        with rec.patch(openrouter_mod):
            res = self._client().generate("sys", "same character?", images=[RED, GREEN])
        self.assertEqual(res.content, "same knight")
        req = rec.requests[0]
        self.assertEqual(str(req.url), f"{BASE}/chat/completions")
        self.assertEqual(req.headers["authorization"], "Bearer or-test")
        self.assertEqual(rec.body["model"], "anthropic/claude-test")
        self.assertEqual(
            [p["type"] for p in rec.body["messages"][1]["content"]],
            ["image_url", "image_url", "text"],
        )

    def test_stream_skips_keepalive_comments_and_names_the_provider(self):
        rec = Recorder(
            sse_lines=[
                ": OPENROUTER PROCESSING",
                'data: {"choices":[{"delta":{"content":"hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"}}]}',
                "data: [DONE]",
            ]
        )
        with rec.patch(openrouter_mod):
            events = list(self._client().stream("sys", "hi"))
        self.assertEqual(events[0]["provider"], "openrouter")
        self.assertEqual(events[-1]["content"], "hello")


class TestOpenRouterImages(unittest.TestCase):
    def _client(self):
        return OpenRouter(api_key="or-test", image_model="openai/gpt-image-test")

    def _response(self, data=BLUE, **extra):
        return {
            "created": 1,
            "data": [{"b64_json": _b64(data), "media_type": "image/png"}],
            **extra,
        }

    def test_text_to_image(self):
        rec = Recorder(
            json_body=self._response(usage={"cost": 0.045, "total_tokens": 9})
        )
        with rec.patch(openrouter_mod):
            pic = self._client().generate_image(
                "a potion icon", aspect_ratio="1:1", background="transparent"
            )
        self.assertEqual(str(rec.requests[0].url), f"{BASE}/images")
        self.assertEqual(
            rec.body,
            {
                "model": "openai/gpt-image-test",
                "prompt": "a potion icon",
                "aspect_ratio": "1:1",
                "background": "transparent",
            },
        )
        self.assertEqual(pic.bytes(), BLUE)
        self.assertEqual(pic.metadata["provider"], "openrouter")
        self.assertEqual(pic.metadata["mode"], "txt2img")
        self.assertEqual(pic.metadata["cost_usd"], 0.045)

    def test_references_go_to_the_same_endpoint_in_order(self):
        rec = Recorder(json_body=self._response())
        with rec.patch(openrouter_mod):
            pic = self._client().generate_image(
                "image 1 with the palette of image 2",
                image=[RED, Picture.from_bytes(GREEN)],
                model="google/gemini-image-test",
            )
        self.assertEqual(str(rec.requests[0].url), f"{BASE}/images")
        self.assertEqual(rec.body["model"], "google/gemini-image-test")
        self.assertEqual(
            [_decode_ref(r) for r in rec.body["input_references"]], [RED, GREEN]
        )
        self.assertNotIn("images", rec.body)
        self.assertEqual(pic.metadata["mode"], "img2img")
        self.assertEqual(pic.metadata["references"], 2)
        self.assertNotIn("cost_usd", pic.metadata)

    def test_mask_is_rejected_with_a_way_forward(self):
        with self.assertRaises(ValueError) as ctx:
            self._client().generate_image("edit", image=RED, mask=GREEN)
        self.assertIn("reference image", str(ctx.exception))

    def test_reference_limit(self):
        with self.assertRaises(ValueError):
            self._client().generate_image("x", image=[RED] * 17)

    def test_error_status(self):
        rec = Recorder(
            json_body={"error": {"message": "Insufficient credits"}}, status=402
        )
        with rec.patch(openrouter_mod):
            with self.assertRaises(RuntimeError) as ctx:
                self._client().generate_image("x")
        self.assertIn("openrouter /images failed: HTTP 402", str(ctx.exception))
        self.assertIn("Insufficient credits", str(ctx.exception))

    def test_stream(self):
        rec = Recorder(json_body=self._response())
        with rec.patch(openrouter_mod):
            events = list(self._client().imagestream("x", image=RED))
        self.assertEqual(events[0]["provider"], "openrouter")
        self.assertEqual(events[0]["model"], "openai/gpt-image-test")
        self.assertEqual(events[-1]["image"].bytes(), BLUE)


class TestOpenRouterModelsAndFactory(unittest.TestCase):
    def test_models(self):
        rec = Recorder(
            routes={
                "/images/models": {
                    "data": [
                        {"id": "z/painter"},
                        {"id": "a/painter", "name": "A Painter"},
                    ]
                },
                "/models": {
                    "data": [
                        {
                            "id": "x/chat",
                            "name": "X Chat",
                            "architecture": {"output_modalities": ["text"]},
                        },
                        {
                            "id": "y/image-only",
                            "architecture": {"output_modalities": ["image"]},
                        },
                        {"id": "w/unlabelled"},
                    ]
                },
            }
        )
        with rec.patch(openrouter_mod):
            models = OpenRouter(api_key="or-test").models()
        self.assertEqual([m["id"] for m in models["text"]], ["w/unlabelled", "x/chat"])
        self.assertEqual(models["text"][1]["label"], "X Chat")
        self.assertEqual([m["id"] for m in models["image"]], ["a/painter", "z/painter"])
        self.assertEqual(
            sorted(str(r.url) for r in rec.requests),
            [f"{BASE}/images/models", f"{BASE}/models"],
        )

    def test_factory(self):
        env = {
            "art_PROVIDER": "openrouter",
            "art_API_KEY": "or-test",
            "art_IMAGE_MODEL": "openai/gpt-image-test",
            "judge_PROVIDER": "openrouter",
            "judge_API_KEY": "or-test",
            "judge_MODEL": "anthropic/claude-test",
        }
        art = trivialai.image(trivialai.from_env("art_", env=env))
        judge = trivialai.text(trivialai.from_env("judge_", env=env))
        self.assertIsInstance(art, OpenRouter)
        self.assertEqual(art.image_model, "openai/gpt-image-test")
        self.assertEqual(judge.model, "anthropic/claude-test")
        self.assertIs(trivialai.OpenRouter, OpenRouter)


if __name__ == "__main__":
    unittest.main()
