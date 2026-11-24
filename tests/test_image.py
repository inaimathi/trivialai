import base64
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from _png import make_png

from src.trivialai.image import ImageMixin, Picture, _looks_like_pil_image

REPO_ROOT = Path(__file__).resolve().parent.parent


PNG = make_png()


class TestPictureWithoutPillow(unittest.TestCase):
    """Bytes-backed Pictures must not need Pillow (it is an optional extra)."""

    def test_bytes_backed_picture_works_with_pillow_hidden(self):
        # Run in a subprocess: hiding Pillow means tampering with the import
        # system, which must not leak into the other test modules.
        script = textwrap.dedent("""
            import builtins, sys
            real_import = builtins.__import__
            def no_pil(name, *a, **k):
                if name == "PIL" or name.startswith("PIL."):
                    raise ImportError("Pillow is hidden for this test")
                return real_import(name, *a, **k)
            builtins.__import__ = no_pil

            from src.trivialai.image import Picture, _looks_like_pil_image
            data = bytes.fromhex(sys.argv[1])
            pic = Picture.from_bytes(data)
            assert pic.media_type == "image/png", pic.media_type
            assert pic.bytes() == data
            assert pic.b64()
            assert not _looks_like_pil_image(object())
            try:
                pic.pil_image()
            except ImportError as e:
                assert "trivialai[image]" in str(e), e
            else:
                raise AssertionError("pil_image() should need Pillow")
            assert "PIL" not in sys.modules
            print("ok")
            """)
        res = subprocess.run(
            [sys.executable, "-c", script, PNG.hex()],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "ok")


class TestPictureWithPillow(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")

    def test_pil_roundtrip(self):
        from PIL import Image

        pic = Picture.from_bytes(PNG)
        img = pic.pil_image()
        self.assertEqual(img.size, (1, 1))
        self.assertTrue(_looks_like_pil_image(img))
        self.assertFalse(_looks_like_pil_image(PNG))

        back = Picture.of(Image.new("RGBA", (2, 3)))
        self.assertEqual(back.media_type, "image/png")
        self.assertEqual(back.pil_image().size, (2, 3))


# ---------------------------------------------------------------------------
# Multiple reference images
# ---------------------------------------------------------------------------

RED = make_png(rgba=(255, 0, 0, 255))
GREEN = make_png(rgba=(0, 255, 0, 255))
BLUE = make_png(rgba=(0, 0, 255, 255))


class _Plain(ImageMixin):
    pass


class _OnlyTwo(ImageMixin):
    MAX_REFERENCE_IMAGES = 2


class TestCoerceImageInputs(unittest.TestCase):
    def test_none_and_empty_mean_txt2img(self):
        m = _Plain()
        self.assertEqual(m._coerce_image_inputs(None), [])
        self.assertEqual(m._coerce_image_inputs([]), [])
        self.assertEqual(m._coerce_image_inputs([None, None]), [])
        self.assertFalse(m._has_image_input(None))
        self.assertFalse(m._has_image_input([]))
        self.assertFalse(m._has_image_input([None]))
        self.assertTrue(m._has_image_input(RED))
        self.assertTrue(m._has_image_input([None, RED]))

    def test_single_image_is_a_one_element_list(self):
        out = _Plain()._coerce_image_inputs(RED)
        self.assertEqual([p.bytes() for p in out], [RED])

    def test_order_and_mixed_kinds_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "green.png")
            with open(path, "wb") as f:
                f.write(GREEN)
            out = _Plain()._coerce_image_inputs(
                [RED, None, path, Picture.from_bytes(BLUE)]
            )
            self.assertEqual([p.bytes() for p in out], [RED, GREEN, BLUE])

    def test_tuple_is_accepted(self):
        out = _Plain()._coerce_image_inputs((RED, BLUE))
        self.assertEqual([p.bytes() for p in out], [RED, BLUE])

    def test_url_is_fetched(self):
        fetched = Picture.from_bytes(GREEN)
        with patch.object(Picture, "from_url", return_value=fetched) as from_url:
            out = _Plain()._coerce_image_inputs([RED, "https://example.com/g.png"])
        from_url.assert_called_once_with("https://example.com/g.png")
        self.assertEqual([p.bytes() for p in out], [RED, GREEN])

    def test_unreadable_input_raises_instead_of_dropping_it(self):
        with self.assertRaises(TypeError):
            _Plain()._coerce_image_inputs([RED, 42])
        with self.assertRaises(TypeError):
            _Plain()._coerce_image_inputs(42)

    def test_class_limit(self):
        m = _OnlyTwo()
        self.assertEqual(len(m._coerce_image_inputs([RED, GREEN])), 2)
        with self.assertRaises(ValueError) as ctx:
            m._coerce_image_inputs([RED, GREEN, BLUE])
        self.assertIn("at most 2", str(ctx.exception))
        self.assertIn("got 3", str(ctx.exception))

    def test_explicit_limit_overrides_class_limit(self):
        m = _OnlyTwo()
        self.assertEqual(len(m._coerce_image_inputs([RED, GREEN, BLUE], limit=None)), 3)
        with self.assertRaises(ValueError) as ctx:
            m._coerce_image_inputs([RED, GREEN], limit=1, what="model-x")
        self.assertIn("model-x", str(ctx.exception))


class TestStabDiffReferences(unittest.TestCase):
    def _sd(self):
        from src.trivialai.stabdiff import StabDiff

        return StabDiff(skip_healthcheck=True)

    def test_txt2img_without_image(self):
        endpoint, payload = self._sd()._build_generation_request("a fox")
        self.assertEqual(endpoint, "/sdapi/v1/txt2img")
        self.assertNotIn("init_images", payload)

    def test_single_image_alone_or_in_a_list(self):
        for image in (RED, [RED]):
            endpoint, payload = self._sd()._build_generation_request(
                "a fox", image=image
            )
            self.assertEqual(endpoint, "/sdapi/v1/img2img")
            self.assertEqual(
                [base64.b64decode(b) for b in payload["init_images"]], [RED]
            )

    def test_several_images_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._sd()._build_generation_request("a fox", image=[RED, GREEN])
        self.assertIn("single init image", str(ctx.exception))


class TestBedrockReferences(unittest.TestCase):
    def setUp(self):
        try:
            from src.trivialai import bedrock
        except ImportError as e:
            self.skipTest(f"bedrock extra not installed: {e}")
        self.bedrock = bedrock

    def _pics(self, *datas):
        return [Picture.from_bytes(d) for d in datas]

    def test_nova_sends_all_references_in_order(self):
        payload = self.bedrock._build_image_payload(
            "amazon.nova-canvas-v1:0", "blend", self._pics(RED, GREEN, BLUE)
        )
        self.assertEqual(payload["taskType"], "IMAGE_VARIATION")
        sent = [base64.b64decode(b) for b in payload["imageVariationParams"]["images"]]
        self.assertEqual(sent, [RED, GREEN, BLUE])

    def test_single_picture_and_none_still_work(self):
        one = self.bedrock._build_image_payload(
            "amazon.nova-canvas-v1:0", "vary", Picture.from_bytes(RED)
        )
        self.assertEqual(len(one["imageVariationParams"]["images"]), 1)
        none = self.bedrock._build_image_payload(
            "amazon.nova-canvas-v1:0", "draw", None
        )
        self.assertEqual(none["taskType"], "TEXT_IMAGE")

    def test_nova_rejects_more_than_five(self):
        with self.assertRaises(ValueError):
            self.bedrock._build_image_payload(
                "amazon.nova-canvas-v1:0", "blend", self._pics(*[RED] * 6)
            )

    def test_stability_takes_exactly_one(self):
        payload = self.bedrock._build_image_payload(
            "stability.stable-diffusion-xl-v1", "edit", self._pics(RED)
        )
        self.assertEqual(base64.b64decode(payload["init_image"]), RED)
        with self.assertRaises(ValueError) as ctx:
            self.bedrock._build_image_payload(
                "stability.stable-diffusion-xl-v1", "edit", self._pics(RED, GREEN)
            )
        self.assertIn("at most 1", str(ctx.exception))

    def test_generate_image_passes_references_to_invoke_model(self):
        with patch("boto3.Session") as MockSession:
            runtime = MagicMock(name="bedrock-runtime")
            control = MagicMock(name="bedrock-control")
            control.list_inference_profiles.return_value = {
                "inferenceProfileSummaries": []
            }
            MockSession.return_value.client.side_effect = lambda svc, **kw: (
                runtime if "runtime" in svc else control
            )
            b = self.bedrock.Bedrock(
                image_model_id="amazon.nova-canvas-v1:0", region="us-east-1"
            )
        body = MagicMock()
        body.read.return_value = json.dumps(
            {"images": [base64.b64encode(BLUE).decode()]}
        ).encode()
        runtime.invoke_model.return_value = {"body": body}

        out = b.generate_image("blend these", image=[RED, GREEN])

        self.assertEqual(out.bytes(), BLUE)
        self.assertEqual(out.metadata["mode"], "img2img")
        sent = json.loads(runtime.invoke_model.call_args.kwargs["body"])
        self.assertEqual(
            [base64.b64decode(x) for x in sent["imageVariationParams"]["images"]],
            [RED, GREEN],
        )


class TestGeminiReferences(unittest.TestCase):
    def setUp(self):
        try:
            from src.trivialai import gemini
        except ImportError as e:
            self.skipTest(f"gemini extra not installed: {e}")
        self.gemini = gemini

    def _client(self):
        g = self.gemini.Gemini(api_key="test-key", image_model="img-model")
        g._client = MagicMock(name="genai-client")
        part = MagicMock()
        part.inline_data.data = BLUE
        part.inline_data.mime_type = "image/png"
        g._client.models.generate_content.return_value = MagicMock(parts=[part])
        return g

    def _sent_parts(self, g):
        contents = g._client.models.generate_content.call_args.kwargs["contents"]
        images = [c.inline_data.data for c in contents[:-1]]
        return images, contents[-1]

    def test_txt2img_sends_only_the_prompt(self):
        g = self._client()
        out = g.generate_image("a fox")
        self.assertEqual(
            g._client.models.generate_content.call_args.kwargs["contents"], ["a fox"]
        )
        self.assertEqual(out.metadata["mode"], "txt2img")

    def test_single_image_precedes_the_prompt(self):
        g = self._client()
        g.generate_image("make it blue", image=RED)
        images, prompt = self._sent_parts(g)
        self.assertEqual(images, [RED])
        self.assertEqual(prompt, "make it blue")

    def test_references_are_sent_in_order_before_the_prompt(self):
        g = self._client()
        out = g.generate_image(
            "image 1 in the palette of image 2", image=[RED, GREEN, BLUE]
        )
        images, prompt = self._sent_parts(g)
        self.assertEqual(images, [RED, GREEN, BLUE])
        self.assertEqual(prompt, "image 1 in the palette of image 2")
        self.assertEqual(out.bytes(), BLUE)
        self.assertEqual(out.metadata["mode"], "img2img")

    def test_stream_reports_mode_and_returns_the_image(self):
        g = self._client()
        events = list(g.imagestream("combine", image=[RED, GREEN]))
        self.assertEqual(events[0]["mode"], "img2img")
        self.assertEqual(events[-1]["type"], "end")
        self.assertEqual(events[-1]["image"].bytes(), BLUE)
        images, _ = self._sent_parts(g)
        self.assertEqual(images, [RED, GREEN])

        g2 = self._client()
        self.assertEqual(list(g2.imagestream("plain", image=[]))[0]["mode"], "txt2img")


if __name__ == "__main__":
    unittest.main()
