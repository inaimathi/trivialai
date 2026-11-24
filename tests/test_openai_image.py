"""
Image generation through the ChatGPT client (OpenAI Images API).

No network: see _http.Recorder.
"""

import asyncio
import base64
import unittest
from unittest.mock import patch

from _http import Recorder
from _png import make_png

import src.trivialai as trivialai
from src.trivialai import chatgpt as chatgpt_mod
from src.trivialai.chatgpt import ChatGPT
from src.trivialai.image import Picture

RED = make_png(rgba=(255, 0, 0, 255))
GREEN = make_png(rgba=(0, 255, 0, 255))
BLUE = make_png(rgba=(0, 0, 255, 255))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _decode(part: dict) -> bytes:
    url = part["image_url"]
    assert url.startswith("data:image/png;base64,"), url[:40]
    return base64.b64decode(url.split(",", 1)[1])


def _images_response(*datas: bytes, **extra) -> dict:
    return {"created": 1, "data": [{"b64_json": _b64(d)} for d in datas], **extra}


class TestOpenAIImages(unittest.TestCase):
    def _client(self, **kwargs):
        kwargs.setdefault("image_model", "gpt-image-test")
        return ChatGPT(api_key="sk-test", **kwargs)

    def test_text_to_image(self):
        rec = Recorder(json_body=_images_response(BLUE, usage={"total_tokens": 42}))
        with rec.patch(chatgpt_mod):
            pic = self._client().generate_image(
                "a blue square", background="transparent", size="1024x1024"
            )

        req = rec.requests[0]
        self.assertEqual(str(req.url), "https://api.openai.com/v1/images/generations")
        self.assertEqual(req.headers["authorization"], "Bearer sk-test")
        self.assertEqual(
            rec.body,
            {
                "model": "gpt-image-test",
                "prompt": "a blue square",
                "background": "transparent",
                "size": "1024x1024",
            },
        )
        self.assertIsInstance(pic, Picture)
        self.assertEqual(pic.bytes(), BLUE)
        self.assertEqual(pic.media_type, "image/png")
        self.assertEqual(pic.metadata["provider"], "openai")
        self.assertEqual(pic.metadata["model"], "gpt-image-test")
        self.assertEqual(pic.metadata["mode"], "txt2img")
        self.assertEqual(pic.metadata["references"], 0)
        self.assertEqual(pic.metadata["usage"], {"total_tokens": 42})

    def test_single_image_edit(self):
        rec = Recorder(json_body=_images_response(BLUE))
        with rec.patch(chatgpt_mod):
            pic = self._client().generate_image("make it blue", image=RED)
        self.assertEqual(
            str(rec.requests[0].url), "https://api.openai.com/v1/images/edits"
        )
        self.assertEqual([_decode(p) for p in rec.body["images"]], [RED])
        self.assertNotIn("mask", rec.body)
        self.assertEqual(pic.metadata["mode"], "img2img")
        self.assertEqual(pic.metadata["references"], 1)

    def test_several_references_are_sent_in_order(self):
        rec = Recorder(json_body=_images_response(BLUE))
        with rec.patch(chatgpt_mod):
            pic = self._client().generate_image(
                "image 1 in the pose of image 2",
                image=[RED, Picture.from_bytes(GREEN)],
                quality="low",
            )
        self.assertEqual([_decode(p) for p in rec.body["images"]], [RED, GREEN])
        self.assertEqual(rec.body["prompt"], "image 1 in the pose of image 2")
        self.assertEqual(rec.body["quality"], "low")
        self.assertEqual(pic.metadata["references"], 2)

    def test_mask_is_sent_with_edits_only(self):
        rec = Recorder(json_body=_images_response(BLUE))
        with rec.patch(chatgpt_mod):
            self._client().generate_image("recolor the cape", image=RED, mask=GREEN)
        self.assertEqual(_decode(rec.body["mask"]), GREEN)

        with self.assertRaises(ValueError):
            self._client().generate_image("no input image", mask=GREEN)

    def test_more_than_sixteen_references_is_rejected_before_any_request(self):
        rec = Recorder(json_body=_images_response(BLUE))
        with rec.patch(chatgpt_mod):
            with self.assertRaises(ValueError) as ctx:
                self._client().generate_image("too many", image=[RED] * 17)
        self.assertIn("at most 16", str(ctx.exception))
        self.assertEqual(rec.requests, [])

    def test_model_override_and_image_index(self):
        rec = Recorder(json_body=_images_response(RED, GREEN))
        with rec.patch(chatgpt_mod):
            pic = self._client().generate_image(
                "two please", model="other-model", n=2, image_index=1
            )
            with self.assertRaises(IndexError):
                self._client().generate_image("two please", n=2, image_index=5)
        self.assertEqual(pic.bytes(), GREEN)
        self.assertEqual(pic.metadata["num_images"], 2)

    def test_image_index_is_not_sent_to_the_api(self):
        rec = Recorder(json_body=_images_response(RED, GREEN))
        with rec.patch(chatgpt_mod):
            self._client().generate_image(
                "two", model="other-model", n=2, image_index=1
            )
        self.assertEqual(rec.body, {"model": "other-model", "prompt": "two", "n": 2})

    def test_missing_image_model_is_loud(self):
        with self.assertRaises(ValueError) as ctx:
            ChatGPT(model="gpt-test", api_key="sk-test").generate_image("x")
        self.assertIn("image_model", str(ctx.exception))

    def test_http_error_carries_status_and_body(self):
        rec = Recorder(json_body={"error": {"message": "bad size"}}, status=400)
        with rec.patch(chatgpt_mod):
            with self.assertRaises(RuntimeError) as ctx:
                self._client().generate_image("x", size="1x1")
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertIn("bad size", str(ctx.exception))

    def test_empty_response_is_an_error(self):
        rec = Recorder(json_body={"created": 1, "data": []})
        with rec.patch(chatgpt_mod):
            with self.assertRaises(RuntimeError) as ctx:
                self._client().generate_image("x")
        self.assertIn("no images", str(ctx.exception))

    def test_legacy_url_response_is_downloaded(self):
        rec = Recorder(json_body={"data": [{"url": "https://cdn.example/x.png"}]})
        with (
            rec.patch(chatgpt_mod),
            patch.object(
                Picture, "from_url", return_value=Picture.from_bytes(GREEN)
            ) as from_url,
        ):
            pic = self._client().generate_image("x")
        from_url.assert_called_once_with("https://cdn.example/x.png")
        self.assertEqual(pic.bytes(), GREEN)

    def test_svg_output_is_labelled(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        rec = Recorder(json_body=_images_response(svg))
        with rec.patch(chatgpt_mod):
            pic = self._client().generate_image("an icon", output_format="svg")
        self.assertEqual(pic.bytes(), svg)
        self.assertEqual(pic.media_type, "image/svg+xml")
        self.assertTrue(pic.file().endswith(".svg"))

    def test_base_url(self):
        rec = Recorder(json_body=_images_response(BLUE))
        with rec.patch(chatgpt_mod):
            self._client(base_url="http://localhost:8000/v1/").generate_image("x")
        self.assertEqual(
            str(rec.requests[0].url), "http://localhost:8000/v1/images/generations"
        )

    def test_imagen_and_stream(self):
        rec = Recorder(json_body=_images_response(BLUE))
        with rec.patch(chatgpt_mod):
            c = self._client(model="gpt-text")
            self.assertEqual(c.imagen("x").bytes(), BLUE)
            events = list(c.imagestream("combine", image=[RED, GREEN]))
            self.assertEqual(asyncio.run(c.aimagen("x")).bytes(), BLUE)

        self.assertEqual([e["type"] for e in events], ["start", "progress", "end"])
        self.assertEqual(events[0]["provider"], "openai")
        # The image model, not the text model.
        self.assertEqual(events[0]["model"], "gpt-image-test")
        self.assertEqual(events[0]["mode"], "img2img")
        self.assertEqual(events[-1]["image"].bytes(), BLUE)

    def test_stream_reports_failures_as_error_events(self):
        rec = Recorder(json_body={"error": {"message": "nope"}}, status=500)
        with rec.patch(chatgpt_mod):
            events = list(self._client().imagestream("x"))
            with self.assertRaises(RuntimeError):
                asyncio.run(self._client().aimagen("x"))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("HTTP 500", events[-1]["message"])


class TestOpenAIModelsAndFactory(unittest.TestCase):
    def test_models_lists_text_and_image(self):
        rec = Recorder(
            json_body={
                "data": [
                    {"id": "gpt-5"},
                    {"id": "gpt-image-2"},
                    {"id": "chatgpt-image-latest"},
                    {"id": "dall-e-3"},
                    {"id": "text-embedding-3-small"},
                    {"id": "whisper-1"},
                    {"id": "o4-mini"},
                ]
            }
        )
        with rec.patch(chatgpt_mod):
            models = ChatGPT(api_key="sk-test").models()
        self.assertEqual([m["id"] for m in models["text"]], ["gpt-5", "o4-mini"])
        self.assertEqual(
            [m["id"] for m in models["image"]],
            ["chatgpt-image-latest", "dall-e-3", "gpt-image-2"],
        )
        self.assertEqual(str(rec.requests[0].url), "https://api.openai.com/v1/models")

    def test_factory_builds_an_image_client(self):
        for provider in ("openai", "chatgpt"):
            m = trivialai.image(
                {
                    "provider": provider,
                    "api_key": "sk-test",
                    "image_model": "gpt-image-test",
                }
            )
            self.assertIsInstance(m, ChatGPT)
            self.assertEqual(m.image_model, "gpt-image-test")

    def test_factory_from_env(self):
        env = {
            "sprite_PROVIDER": "openai",
            "sprite_API_KEY": "sk-test",
            "sprite_IMAGE_MODEL": "gpt-image-test",
            "sprite_BASE_URL": "http://localhost:9/v1",
            "sprite_TIMEOUT": "12.5",
        }
        m = trivialai.image(trivialai.from_env("sprite_", env=env))
        self.assertEqual(m.image_model, "gpt-image-test")
        self.assertEqual(m.base_url, "http://localhost:9/v1")
        self.assertEqual(m.timeout, 12.5)

    def test_text_still_works_and_uses_base_url(self):
        rec = Recorder(json_body={"choices": [{"message": {"content": "hi"}}]})
        with rec.patch(chatgpt_mod):
            m = trivialai.text(
                {
                    "provider": "openai",
                    "api_key": "sk-test",
                    "model": "gpt-test",
                    "base_url": "http://localhost:9/v1",
                }
            )
            self.assertEqual(m.generate("s", "p").content, "hi")
        self.assertEqual(
            str(rec.requests[0].url), "http://localhost:9/v1/chat/completions"
        )


if __name__ == "__main__":
    unittest.main()
