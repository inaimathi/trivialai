import httpx
from src.trivialai.deepseek import DeepSeek


def test_deepseek_models(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {
                "object": "list",
                "data": [
                    {
                        "id": "deepseek-flash",
                        "object": "model",
                        "owned_by": "deepseek",
                    },
                    {
                        "id": "deepseek-v4-pro",
                        "object": "model",
                        "owned_by": "deepseek",
                    },
                ],
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, headers=None):
            assert url == "https://api.deepseek.com/models"
            assert headers["Authorization"] == "Bearer test-key"
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)

    client = DeepSeek(api_key="test-key")

    assert client.models() == [
        {
            "id": "deepseek-flash",
            "object": "model",
            "owned_by": "deepseek",
        },
        {
            "id": "deepseek-v4-pro",
            "object": "model",
            "owned_by": "deepseek",
        },
    ]
    assert client.model_names() == ["deepseek-flash", "deepseek-v4-pro"]


def test_deepseek_models_http_error(monkeypatch):
    class Response:
        status_code = 401

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)

    client = DeepSeek(api_key="bad-key")

    try:
        client.models()
    except ValueError as exc:
        assert "HTTP 401" in str(exc)
    else:
        raise AssertionError("DeepSeek model discovery accepted an HTTP error")
