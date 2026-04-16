# tests/test_bedrock.py
"""
Tests for the Amazon Bedrock adapter (src/trivialai/bedrock.py).

Strategy
--------
boto3.Session is patched at construction time so no AWS credentials or network
access are needed.  The _make_bedrock() helper returns the Bedrock instance
together with the two named MagicMocks that were injected as _runtime and
_control, allowing per-test response configuration and call-argument
inspection.
"""
import asyncio
import unittest
from unittest.mock import MagicMock, call, patch

from botocore.exceptions import ClientError
from src.trivialai.bedrock import (_DEFAULT_TEXT_MODEL, Bedrock,
                                   _region_to_geo_prefix)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _client_error(code: str, message: str = "simulated") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "operation")


def _make_bedrock(**kwargs):
    """
    Construct a Bedrock instance with boto3.Session fully mocked.

    Returns (bedrock, mock_runtime, mock_control).  The same mock objects are
    used both *during* construction (so event registrations are captured) and
    available afterward for configuring return values and asserting calls.
    """
    with patch("boto3.Session") as MockSession:
        mock_runtime = MagicMock(name="bedrock-runtime")
        mock_control = MagicMock(name="bedrock-control")
        MockSession.return_value.client.side_effect = lambda svc, **kw: (
            mock_runtime if "runtime" in svc else mock_control
        )
        b = Bedrock(**kwargs)
    return b, mock_runtime, mock_control


def _converse_response(text: str) -> dict:
    return {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {"message": {"content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


def _stream_response(*chunks: str, output_tokens: int = 5) -> dict:
    """Fake converse_stream() return value."""
    events = [{"contentBlockDelta": {"delta": {"text": chunk}}} for chunk in chunks]
    events.append({"metadata": {"usage": {"outputTokens": output_tokens}}})
    return {"stream": iter(events)}


# ---------------------------------------------------------------------------
# _region_to_geo_prefix unit tests
# ---------------------------------------------------------------------------


class TestRegionToGeoPrefix(unittest.TestCase):

    def test_us_regions(self):
        self.assertEqual(_region_to_geo_prefix("us-east-1"), "us")
        self.assertEqual(_region_to_geo_prefix("us-west-2"), "us")

    def test_eu_regions(self):
        self.assertEqual(_region_to_geo_prefix("eu-west-1"), "eu")
        self.assertEqual(_region_to_geo_prefix("eu-central-1"), "eu")

    def test_ap_regions(self):
        self.assertEqual(_region_to_geo_prefix("ap-southeast-1"), "ap")
        self.assertEqual(_region_to_geo_prefix("ap-northeast-1"), "ap")

    def test_unsupported_regions_return_none(self):
        for region in ("ca-central-1", "sa-east-1", "me-south-1", "af-south-1"):
            with self.subTest(region=region):
                self.assertIsNone(_region_to_geo_prefix(region))


# ---------------------------------------------------------------------------
# Constructor / model_id resolution (the design-table cases)
# ---------------------------------------------------------------------------


class TestBedrockConstructor(unittest.TestCase):
    """
    Enforces the model_id × region decision table:

      model_id  | region        | result
      ──────────┼───────────────┼───────────────────────────────────────────
      None      | us-* eu-* ap-* | f"{prefix}.{DEFAULT_MODEL}"
      None      | ca-* sa-* …   | None  (discovery-only)
      bare ID   | supported     | f"{prefix}.{model_id}"
      prefixed  | matches       | unchanged
      explicit  | unsupported   | ValueError
      prefixed  | wrong prefix  | ValueError
    """

    def test_none_us_region_applies_us_prefix(self):
        b, _, _ = _make_bedrock(region="us-east-1")
        self.assertEqual(b._model_id, f"us.{_DEFAULT_TEXT_MODEL}")

    def test_none_eu_region_applies_eu_prefix(self):
        b, _, _ = _make_bedrock(region="eu-west-1")
        self.assertEqual(b._model_id, f"eu.{_DEFAULT_TEXT_MODEL}")

    def test_none_ap_region_applies_ap_prefix(self):
        b, _, _ = _make_bedrock(region="ap-southeast-1")
        self.assertEqual(b._model_id, f"ap.{_DEFAULT_TEXT_MODEL}")

    def test_none_unsupported_region_gives_discovery_only(self):
        b, _, _ = _make_bedrock(region="ca-central-1")
        self.assertIsNone(b._model_id)

    def test_bare_id_gets_prefix_from_region(self):
        b, _, _ = _make_bedrock(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.assertEqual(b._model_id, "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    def test_correct_prefixed_id_is_unchanged(self):
        mid = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        b, _, _ = _make_bedrock(model_id=mid, region="us-east-1")
        self.assertEqual(b._model_id, mid)

    def test_explicit_id_unsupported_region_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_bedrock(
                model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
                region="ca-central-1",
            )
        self.assertIn("ca-central-1", str(ctx.exception))

    def test_explicit_prefixed_id_unsupported_region_raises(self):
        # Even an already-prefixed ID is rejected in unsupported regions.
        with self.assertRaises(ValueError):
            _make_bedrock(
                model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
                region="ca-central-1",
            )

    def test_prefix_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_bedrock(
                model_id="eu.anthropic.claude-3-5-haiku-20241022-v1:0",
                region="us-east-1",
            )
        msg = str(ctx.exception)
        self.assertIn("eu", msg)
        self.assertIn("us", msg)

    def test_model_id_is_private(self):
        b, _, _ = _make_bedrock()
        self.assertFalse(
            hasattr(b, "model_id"),
            "model_id should be private (_model_id); no public attribute expected",
        )


# ---------------------------------------------------------------------------
# Bearer-token authentication
# ---------------------------------------------------------------------------


class TestBedrockBearerToken(unittest.TestCase):

    TOKEN = "test-bearer-token-xyz"

    def setUp(self):
        self.b, self.mock_runtime, self.mock_control = _make_bedrock(
            aws_bearer_token=self.TOKEN
        )

    def _registered_handlers(self, mock_client):
        """Return {event_name: handler} for all register() calls on a client."""
        return {
            args[0]: args[1]
            for args, _ in mock_client.meta.events.register.call_args_list
        }

    def test_before_sign_registered_on_runtime(self):
        handlers = self._registered_handlers(self.mock_runtime)
        self.assertIn("before-sign.bedrock-runtime.*", handlers)

    def test_before_sign_registered_on_control(self):
        handlers = self._registered_handlers(self.mock_control)
        self.assertIn("before-sign.bedrock.*", handlers)

    def test_handler_injects_authorization_header(self):
        handlers = self._registered_handlers(self.mock_runtime)
        handler = handlers["before-sign.bedrock-runtime.*"]

        fake_request = MagicMock()
        fake_request.headers = {}
        handler(fake_request)

        self.assertEqual(
            fake_request.headers["Authorization"],
            f"Bearer {self.TOKEN}",
        )

    def test_handler_uses_exact_token_string(self):
        # A second instance with a different token should get its own closure.
        b2, rt2, _ = _make_bedrock(aws_bearer_token="other-token")
        handlers2 = {
            args[0]: args[1] for args, _ in rt2.meta.events.register.call_args_list
        }
        handler2 = handlers2["before-sign.bedrock-runtime.*"]
        req = MagicMock()
        req.headers = {}
        handler2(req)
        self.assertEqual(req.headers["Authorization"], "Bearer other-token")


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestBedrockGenerate(unittest.TestCase):

    def setUp(self):
        self.b, self.mock_runtime, _ = _make_bedrock()

    def test_returns_content_from_response(self):
        self.mock_runtime.converse.return_value = _converse_response("Hello world")
        result = self.b.generate("system", "prompt")
        self.assertEqual(result.content, "Hello world")

    def test_strips_whitespace_from_content(self):
        self.mock_runtime.converse.return_value = _converse_response("  hi  ")
        result = self.b.generate("sys", "prompt")
        self.assertEqual(result.content, "hi")

    def test_passes_system_prompt_to_converse(self):
        self.mock_runtime.converse.return_value = _converse_response("ok")
        self.b.generate("be helpful", "what time is it?")
        kw = self.mock_runtime.converse.call_args.kwargs
        self.assertEqual(kw["system"], [{"text": "be helpful"}])

    def test_passes_user_message_to_converse(self):
        self.mock_runtime.converse.return_value = _converse_response("ok")
        self.b.generate("sys", "what time is it?")
        kw = self.mock_runtime.converse.call_args.kwargs
        self.assertEqual(
            kw["messages"],
            [{"role": "user", "content": [{"text": "what time is it?"}]}],
        )

    def test_passes_model_id_to_converse(self):
        self.mock_runtime.converse.return_value = _converse_response("ok")
        self.b.generate("sys", "prompt")
        kw = self.mock_runtime.converse.call_args.kwargs
        self.assertEqual(kw["modelId"], self.b._model_id)

    def test_client_error_returns_llmresult_with_none_content(self):
        self.mock_runtime.converse.side_effect = _client_error(
            "ResourceNotFoundException"
        )
        result = self.b.generate("sys", "prompt")
        self.assertIsNone(result.content)
        self.assertIsInstance(result.raw, ClientError)

    def test_empty_response_returns_none_content(self):
        # A response with no text blocks should give content=None, not "".
        self.mock_runtime.converse.return_value = {
            "output": {"message": {"content": []}}
        }
        result = self.b.generate("sys", "prompt")
        self.assertIsNone(result.content)

    def test_discovery_only_raises_value_error(self):
        b, _, _ = _make_bedrock(region="ca-central-1")
        self.assertIsNone(b._model_id)
        with self.assertRaises(ValueError) as ctx:
            b.generate("sys", "prompt")
        self.assertIn("discovery-only", str(ctx.exception))


# ---------------------------------------------------------------------------
# astream()
# ---------------------------------------------------------------------------


class TestBedrockAstream(unittest.TestCase):

    def setUp(self):
        self.b, self.mock_runtime, _ = _make_bedrock()

    def _run(self, coro):
        return asyncio.run(coro)

    async def _collect(self):
        return [e async for e in self.b.astream("sys", "prompt")]

    def test_yields_start_delta_end(self):
        self.mock_runtime.converse_stream.return_value = _stream_response(
            "Hello", " world"
        )
        events = self._run(self._collect())
        types = [e["type"] for e in events]
        self.assertIn("start", types)
        self.assertIn("delta", types)
        self.assertIn("end", types)

    def test_start_event_carries_model_id(self):
        self.mock_runtime.converse_stream.return_value = _stream_response("hi")
        events = self._run(self._collect())
        start = next(e for e in events if e["type"] == "start")
        self.assertEqual(start["model"], self.b._model_id)
        self.assertEqual(start["provider"], "bedrock")

    def test_end_event_aggregates_full_content(self):
        self.mock_runtime.converse_stream.return_value = _stream_response(
            "Hello", " world"
        )
        events = self._run(self._collect())
        end = next(e for e in events if e["type"] == "end")
        self.assertEqual(end["content"], "Hello world")

    def test_delta_events_cover_all_chunks(self):
        self.mock_runtime.converse_stream.return_value = _stream_response(
            "one", " two", " three"
        )
        events = self._run(self._collect())
        deltas = [e for e in events if e["type"] == "delta"]
        self.assertEqual("".join(e["text"] for e in deltas), "one two three")

    def test_client_error_yields_error_event(self):
        self.mock_runtime.converse_stream.side_effect = _client_error(
            "ResourceNotFoundException"
        )
        events = self._run(self._collect())
        types = [e["type"] for e in events]
        self.assertIn("error", types)

    def test_throttle_error_yields_error_event_when_retries_disabled(self):
        self.b.retry_on_throttle = False
        self.mock_runtime.converse_stream.side_effect = _client_error(
            "ThrottlingException"
        )
        events = self._run(self._collect())
        types = [e["type"] for e in events]
        self.assertIn("error", types)

    def test_missing_stream_key_yields_error_event(self):
        # converse_stream() succeeds but returns no "stream" key.
        self.mock_runtime.converse_stream.return_value = {}
        events = self._run(self._collect())
        types = [e["type"] for e in events]
        self.assertIn("error", types)

    def test_discovery_only_yields_error_event(self):
        b, mock_rt, _ = _make_bedrock(region="ca-central-1")

        async def collect():
            return [e async for e in b.astream("sys", "prompt")]

        events = self._run(collect())
        types = [e["type"] for e in events]
        self.assertIn("error", types)
        mock_rt.converse_stream.assert_not_called()


# ---------------------------------------------------------------------------
# models() / discovery helpers
# ---------------------------------------------------------------------------


class TestBedrockModels(unittest.TestCase):

    _ACTIVE_FOUNDATION = {
        "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "modelName": "Claude 3.5 Haiku",
        "providerName": "Anthropic",
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "ACTIVE"},
    }
    _LEGACY_FOUNDATION = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "modelName": "Claude 3 Haiku",
        "providerName": "Anthropic",
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "LEGACY"},
    }
    _ACTIVE_PROFILE = {
        "inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "inferenceProfileName": "US Claude 3.5 Haiku",
        "models": [
            {
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0"
            }
        ],
        "status": "ACTIVE",
        "type": "SYSTEM_DEFINED",
    }
    _INACTIVE_PROFILE = {
        "inferenceProfileId": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "inferenceProfileName": "US Claude 3 Haiku",
        "models": [
            {
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
            }
        ],
        "status": "INACTIVE",
        "type": "SYSTEM_DEFINED",
    }

    def setUp(self):
        self.b, _, self.mock_control = _make_bedrock()
        self.mock_control.list_foundation_models.return_value = {
            "modelSummaries": [self._ACTIVE_FOUNDATION, self._LEGACY_FOUNDATION]
        }
        self.mock_control.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [self._ACTIVE_PROFILE, self._INACTIVE_PROFILE]
        }

    def test_active_only_excludes_legacy_foundation_models(self):
        result = self.b.models(active_only=True)
        ids = [m["model_id"] for m in result["text"]]
        self.assertIn("anthropic.claude-3-5-haiku-20241022-v1:0", ids)
        self.assertNotIn("anthropic.claude-3-haiku-20240307-v1:0", ids)

    def test_active_only_false_includes_legacy_foundation_models(self):
        result = self.b.models(active_only=False)
        ids = [m["model_id"] for m in result["text"]]
        self.assertIn("anthropic.claude-3-haiku-20240307-v1:0", ids)

    def test_active_only_excludes_inactive_profiles(self):
        result = self.b.models(active_only=True)
        ids = [p["profile_id"] for p in result["inference_profiles"]]
        self.assertIn("us.anthropic.claude-3-5-haiku-20241022-v1:0", ids)
        self.assertNotIn("us.anthropic.claude-3-haiku-20240307-v1:0", ids)

    def test_active_only_false_includes_inactive_profiles(self):
        result = self.b.models(active_only=False)
        ids = [p["profile_id"] for p in result["inference_profiles"]]
        self.assertIn("us.anthropic.claude-3-haiku-20240307-v1:0", ids)

    def test_inference_profile_ids_returns_active_only_by_default(self):
        ids = self.b.inference_profile_ids()
        self.assertIn("us.anthropic.claude-3-5-haiku-20241022-v1:0", ids)
        self.assertNotIn("us.anthropic.claude-3-haiku-20240307-v1:0", ids)

    def test_text_model_ids_returns_active_only_by_default(self):
        ids = self.b.text_model_ids()
        self.assertIn("anthropic.claude-3-5-haiku-20241022-v1:0", ids)
        self.assertNotIn("anthropic.claude-3-haiku-20240307-v1:0", ids)

    def test_models_returns_inference_profiles_key(self):
        result = self.b.models()
        self.assertIn("inference_profiles", result)

    def test_list_inference_profiles_failure_returns_empty_list(self):
        self.mock_control.list_inference_profiles.side_effect = _client_error(
            "AccessDeniedException"
        )
        with self.assertLogs("src.trivialai.bedrock", level="WARNING") as cm:
            result = self.b.models()
        self.assertEqual(result["inference_profiles"], [])
        self.assertTrue(any("inference profiles" in line for line in cm.output))

    def test_discovery_only_instance_can_still_call_models(self):
        b, _, mock_control = _make_bedrock(region="ca-central-1")
        mock_control.list_foundation_models.return_value = {
            "modelSummaries": [self._ACTIVE_FOUNDATION]
        }
        mock_control.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [self._ACTIVE_PROFILE]
        }
        # Should not raise even though _model_id is None.
        result = b.models()
        self.assertIn("inference_profiles", result)


if __name__ == "__main__":
    unittest.main()
