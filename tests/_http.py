"""
Test fixture: run adapter code against a recorded HTTP exchange.

httpx.Client / httpx.AsyncClient are replaced with clients bound to an
httpx.MockTransport, so the real request-building and response-parsing code
runs with no network.
"""

import json
from unittest.mock import patch

import httpx


class Recorder:
    """An httpx handler that records requests and replies with a canned response."""

    def __init__(self, *, json_body=None, sse_lines=None, status=200, routes=None):
        self.requests = []
        self._json = json_body
        self._sse = sse_lines
        self._status = status
        # Optional {url path suffix: JSON body}, checked longest suffix first.
        self._routes = routes or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for suffix in sorted(self._routes, key=len, reverse=True):
            if request.url.path.endswith(suffix):
                return httpx.Response(self._status, json=self._routes[suffix])
        if self._sse is not None:
            payload = "".join(f"{line}\n\n" for line in self._sse).encode()
            return httpx.Response(
                self._status,
                content=payload,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(self._status, json=self._json)

    @property
    def body(self):
        return json.loads(self.requests[-1].content)

    def patch(self, module):
        """Patch module.httpx so new clients use this handler."""
        real_client, real_async = httpx.Client, httpx.AsyncClient
        transport = httpx.MockTransport(self)

        def client(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        def aclient(*args, **kwargs):
            kwargs["transport"] = transport
            return real_async(*args, **kwargs)

        return patch.multiple(module.httpx, Client=client, AsyncClient=aclient)
