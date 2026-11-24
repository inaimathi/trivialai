"""
Vision input for the Claude and ChatGPT text clients.

No network: see _http.Recorder.
"""

import asyncio
import base64
import os
import tempfile
import unittest

from _http import Recorder
from _png import make_png

from src.trivialai import chatgpt as chatgpt_mod
from src.trivialai import claude as claude_mod
from src.trivialai.chatgpt import ChatGPT
from src.trivialai.claude import Claude
from src.trivialai.image import Picture, encode_image_input

RED = make_png(rgba=(255, 0, 0, 255))
GREEN = make_png(rgba=(0, 255, 0, 255))
JPEG_MAGIC = b"\xff\xd8\xff\xe0" + b"\x00" * 16  # sniffed as JPEG, never decoded


class TestEncodeImageInput(unittest.TestCase):
    def test_bytes_keep_their_format(self):
        media_type, data = encode_image_input(RED)
        self.assertEqual(media_type, "image/png")
        self.assertEqual(base64.b64decode(data), RED)

    def test_media_type_comes_from_the_bytes_not_the_label(self):
        mislabeled = Picture.from_bytes(RED, media_type="image/jpeg")
        self.assertEqual(encode_image_input(mislabeled)[0], "image/png")

    def test_file_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.png")
            with open(path, "wb") as f:
                f.write(GREEN)
            media_type, data = encode_image_input(path)
        self.assertEqual((media_type, base64.b64decode(data)), ("image/png", GREEN))

    def test_data_url(self):
        url = Picture.from_bytes(RED).data_url()
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), RED)

    def test_unsupported_format_is_reencoded_as_png(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        bmp_bytes = Picture.from_pil(
            Image.new("RGB", (2, 2), (0, 0, 255)), fmt="BMP"
        ).bytes()
        self.assertTrue(bmp_bytes.startswith(b"BM"))

        media_type, data = encode_image_input(
            bmp_bytes, allowed={"image/png", "image/jpeg"}
        )
        self.assertEqual(media_type, "image/png")
        out = Picture.from_bytes(base64.b64decode(data)).pil_image()
        self.assertEqual(out.size, (2, 2))
        self.assertEqual(out.convert("RGB").getpixel((0, 0)), (0, 0, 255))

        # Without a restriction the original bytes go through untouched.
        self.assertEqual(encode_image_input(bmp_bytes)[0], "image/bmp")


class TestClaudeVision(unittest.TestCase):
    def _client(self):
        return Claude(model="claude-test", api_key="sk-test", max_tokens=64)

    def test_text_only_body_is_unchanged(self):
        body = self._client()._build_body("sys", "hello")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(body["system"], "sys")
        self.assertNotIn("stream", body)
        self.assertEqual(
            self._client()._build_body("sys", "hello", [])["messages"][0]["content"],
            "hello",
        )

    def test_images_become_ordered_blocks_before_the_prompt(self):
        body = self._client()._build_body("sys", "compare", [RED, JPEG_MAGIC])
        content = body["messages"][0]["content"]
        self.assertEqual([b["type"] for b in content], ["image", "image", "text"])
        self.assertEqual(content[0]["source"]["type"], "base64")
        self.assertEqual(content[0]["source"]["media_type"], "image/png")
        self.assertEqual(base64.b64decode(content[0]["source"]["data"]), RED)
        self.assertEqual(content[1]["source"]["media_type"], "image/jpeg")
        self.assertEqual(content[2], {"type": "text", "text": "compare"})

    def test_generate_sends_images_and_returns_text(self):
        rec = Recorder(
            json_body={"content": [{"type": "text", "text": "a red square"}]}
        )
        with rec.patch(claude_mod):
            res = self._client().generate("sys", "what is this?", images=[RED])
        self.assertEqual(res.content, "a red square")
        self.assertEqual(
            str(rec.requests[0].url), "https://api.anthropic.com/v1/messages"
        )
        content = rec.body["messages"][0]["content"]
        self.assertEqual(base64.b64decode(content[0]["source"]["data"]), RED)
        self.assertEqual(content[-1]["text"], "what is this?")

    def test_generate_json_works(self):
        # generate_checked calls generate(..., images=...); before vision
        # support the sync generate() did not take that argument at all.
        rec = Recorder(
            json_body={"content": [{"type": "text", "text": '{"ok": true}'}]}
        )
        with rec.patch(claude_mod):
            self.assertEqual(
                self._client().generate_json("sys", "json please").content, {"ok": True}
            )
            self.assertEqual(
                self._client().generate_json("sys", "judge", images=[RED]).content,
                {"ok": True},
            )
        self.assertEqual(rec.body["messages"][0]["content"][0]["type"], "image")

    def test_stream_sends_images(self):
        rec = Recorder(
            sse_lines=[
                'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"red "}}',
                'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"square"}}',
                'data: {"type":"message_stop"}',
            ]
        )
        with rec.patch(claude_mod):
            events = list(self._client().stream("sys", "what?", [RED, GREEN]))
        self.assertEqual(events[-1]["type"], "end")
        self.assertEqual(events[-1]["content"], "red square")
        self.assertTrue(rec.body["stream"])
        sent = [
            base64.b64decode(b["source"]["data"])
            for b in rec.body["messages"][0]["content"]
            if b["type"] == "image"
        ]
        self.assertEqual(sent, [RED, GREEN])

    def test_stream_reports_an_unreadable_image_without_calling_the_api(self):
        rec = Recorder(sse_lines=[])
        with rec.patch(claude_mod):
            events = list(self._client().stream("sys", "what?", [12345]))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("could not prepare images", events[-1]["message"])
        self.assertEqual(rec.requests, [])


class TestChatGPTVision(unittest.TestCase):
    def _client(self):
        return ChatGPT(model="gpt-test", api_key="sk-test")

    def test_text_only_body_is_unchanged(self):
        body = self._client()._build_body("sys", "hello")
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertNotIn("stream", body)

    def test_images_become_ordered_data_urls_before_the_prompt(self):
        body = self._client()._build_body("sys", "compare", [RED, JPEG_MAGIC])
        content = body["messages"][1]["content"]
        self.assertEqual(
            [p["type"] for p in content], ["image_url", "image_url", "text"]
        )
        url = content[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), RED)
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        self.assertEqual(content[2], {"type": "text", "text": "compare"})

    def test_generate_sends_images_and_returns_text(self):
        rec = Recorder(
            json_body={"choices": [{"message": {"content": "a red square"}}]}
        )
        with rec.patch(chatgpt_mod):
            res = self._client().generate("sys", "what is this?", images=[RED])
        self.assertEqual(res.content, "a red square")
        self.assertEqual(
            str(rec.requests[0].url), "https://api.openai.com/v1/chat/completions"
        )
        self.assertEqual(rec.body["messages"][1]["content"][0]["type"], "image_url")

    def test_generate_json_works(self):
        rec = Recorder(
            json_body={"choices": [{"message": {"content": '{"score": 7}'}}]}
        )
        with rec.patch(chatgpt_mod):
            self.assertEqual(
                self._client().generate_json("sys", "json").content, {"score": 7}
            )
            self.assertEqual(
                self._client().generate_json("sys", "judge", images=[RED]).content,
                {"score": 7},
            )

    def test_stream_and_agenerate_send_images(self):
        sse = [
            'data: {"choices":[{"delta":{"content":"red "}}]}',
            'data: {"choices":[{"delta":{"content":"square"}}]}',
            "data: [DONE]",
        ]
        rec = Recorder(sse_lines=sse)
        with rec.patch(chatgpt_mod):
            events = list(self._client().stream("sys", "what?", [RED, GREEN]))
        self.assertEqual(events[-1]["content"], "red square")
        self.assertTrue(rec.body["stream"])
        urls = [
            p["image_url"]["url"]
            for p in rec.body["messages"][1]["content"]
            if p["type"] == "image_url"
        ]
        self.assertEqual(
            [base64.b64decode(u.split(",", 1)[1]) for u in urls], [RED, GREEN]
        )

        rec2 = Recorder(sse_lines=sse)
        with rec2.patch(chatgpt_mod):
            res = asyncio.run(self._client().agenerate("sys", "what?", [RED]))
        self.assertEqual(res.content, "red square")
        self.assertEqual(rec2.body["messages"][1]["content"][0]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
