# src/trivialai/stabdiff.py
from __future__ import annotations

import asyncio
import base64
import json
from contextlib import suppress
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .filesystem import FilesystemMixin
from .image import ImageMixin, Picture


def _strip_data_url_prefix(b64: str) -> str:
    """
    Accept either raw base64 or a data URL like:
      data:image/png;base64,AAAA...
    and return the base64 payload only.
    """
    s = (b64 or "").strip()
    if s.startswith("data:") and "," in s:
        return s.split(",", 1)[1]
    return s


def _decode_b64_image_bytes(b64: str) -> bytes:
    return base64.b64decode(_strip_data_url_prefix(b64))


def _encode_b64_image_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _safe_json_loads(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(v)


class StabDiff(ImageMixin, FilesystemMixin):
    """
    AUTOMATIC1111 Stable Diffusion WebUI client (txt2img + img2img via one API).

    Unified interface:
      - imagen(prompt, image=None, ...)
      - imagestream(prompt, image=None, ...)

    If `image` is None:
      - uses /sdapi/v1/txt2img

    If `image` is provided (Picture / bytes / file path / PIL image):
      - uses /sdapi/v1/img2img with init_images=[...]

    Notes
    -----
    - Per-request model selection is applied with `override_settings`
      (`sd_model_checkpoint`) by default, so we avoid mutating global WebUI state.
    - `set_model(..., persist=True)` can change the loaded model globally via
      `/sdapi/v1/options`.
    """

    def __init__(
        self,
        webui_server: str = "http://127.0.0.1:7860",
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        progress_poll_interval: float = 0.5,
        progress_timeout: Optional[float] = 10.0,
        auth: Optional[tuple[str, str]] = None,
        skip_healthcheck: bool = False,
        send_images: bool = True,
        use_override_settings: bool = True,
        include_previews: bool = True,
    ):
        self.server = webui_server.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.progress_poll_interval = max(0.05, float(progress_poll_interval))
        self.progress_timeout = progress_timeout
        self.auth = auth
        self.default_send_images = send_images
        self.use_override_settings = use_override_settings
        self.include_previews = include_previews

        if not skip_healthcheck:
            self._startup_health_check()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _client(self, *, timeout: Optional[float] = None) -> httpx.Client:
        return httpx.Client(timeout=(self.timeout if timeout is None else timeout))

    def _aclient(self, *, timeout: Optional[float] = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=(self.timeout if timeout is None else timeout))

    def _url(self, path: str) -> str:
        return f"{self.server}{path}"

    def _startup_health_check(self) -> None:
        try:
            with self._client(timeout=min(self.timeout or 30.0, 30.0)) as client:
                resp = client.get(self._url("/sdapi/v1/sd-models"), auth=self.auth)
        except httpx.RequestError as e:
            raise ValueError(
                f"Cannot reach Stable Diffusion WebUI at {self.server}: {e}"
            ) from e

        if resp.status_code != 200:
            raise ValueError(
                f"Stable Diffusion WebUI at {self.server} is not responding as an API "
                f"(HTTP {resp.status_code} from /sdapi/v1/sd-models). "
                "Make sure WebUI is started with `--api`."
            )

    # ------------------------------------------------------------------
    # Model discovery / options
    # ------------------------------------------------------------------

    def loras_full(self) -> List[Dict[str, Any]]:
        """
        Returns the raw LoRA entries from `/sdapi/v1/loras` (when available).

        Typical fields include: name, alias, path, metadata (varies by WebUI build).
        """
        with self._client() as client:
            res = client.get(self._url("/sdapi/v1/loras"), auth=self.auth)

        if res.status_code != 200:
            raise RuntimeError(f"loras failed: HTTP {res.status_code}")

        data = res.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected loras response: {type(data).__name__}")
        return [x for x in data if isinstance(x, dict)]

    def loras(self) -> List[str]:
        """
        Returns a user-friendly list of LoRA identifiers, preferring `name` then `alias`.
        """
        out: List[str] = []
        for item in self.loras_full():
            name = item.get("name")
            if isinstance(name, str) and name:
                out.append(name)
                continue
            alias = item.get("alias")
            if isinstance(alias, str) and alias:
                out.append(alias)
                continue
            # Fallback: some builds may use different keys
            fn = item.get("filename") or item.get("file") or item.get("path")
            if isinstance(fn, str) and fn:
                out.append(fn)
        return out

        def models_full(self) -> List[Dict[str, Any]]:
            with self._client() as client:
                res = client.get(self._url("/sdapi/v1/sd-models"), auth=self.auth)

            if res.status_code != 200:
                raise RuntimeError(f"sd-models failed: HTTP {res.status_code}")

            data = res.json()
            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected sd-models response: {type(data).__name__}"
                )
            return [x for x in data if isinstance(x, dict)]

    def models(self) -> List[str]:
        out: List[str] = []
        for item in self.models_full():
            title = item.get("title")
            if isinstance(title, str) and title:
                out.append(title)
                continue
            model_name = item.get("model_name")
            if isinstance(model_name, str) and model_name:
                out.append(model_name)
                continue
            filename = item.get("filename")
            if isinstance(filename, str) and filename:
                out.append(filename)
        return out

    async def amodels_full(self) -> List[Dict[str, Any]]:
        async with self._aclient() as client:
            res = await client.get(self._url("/sdapi/v1/sd-models"), auth=self.auth)

        if res.status_code != 200:
            raise RuntimeError(f"sd-models failed: HTTP {res.status_code}")

        data = res.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected sd-models response: {type(data).__name__}")
        return [x for x in data if isinstance(x, dict)]

    async def amodels(self) -> List[str]:
        items = await self.amodels_full()
        out: List[str] = []
        for item in items:
            title = item.get("title")
            if isinstance(title, str) and title:
                out.append(title)
                continue
            model_name = item.get("model_name")
            if isinstance(model_name, str) and model_name:
                out.append(model_name)
                continue
            filename = item.get("filename")
            if isinstance(filename, str) and filename:
                out.append(filename)
        return out

    def options(self) -> Dict[str, Any]:
        with self._client() as client:
            res = client.get(self._url("/sdapi/v1/options"), auth=self.auth)

        if res.status_code != 200:
            raise RuntimeError(f"options GET failed: HTTP {res.status_code}")

        j = res.json()
        if not isinstance(j, dict):
            raise RuntimeError(f"Unexpected options response: {type(j).__name__}")
        return j

    def set_options(self, **options: Any) -> Dict[str, Any]:
        with self._client() as client:
            res = client.post(
                self._url("/sdapi/v1/options"), json=options, auth=self.auth
            )

        if res.status_code != 200:
            raise RuntimeError(f"options POST failed: HTTP {res.status_code}")

        try:
            j = res.json()
            if isinstance(j, dict):
                return j
        except Exception:
            pass
        return {}

    def set_model(self, model: str, *, persist: bool = True) -> None:
        self.model = model
        if persist:
            self.set_options(sd_model_checkpoint=model)

    # ------------------------------------------------------------------
    # Request shaping
    # ------------------------------------------------------------------

    def _build_generation_request(
        self,
        prompt: str,
        *,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Returns (endpoint, payload), where endpoint is txt2img or img2img
        depending on whether an input image is provided.
        """
        payload: Dict[str, Any] = dict(kwargs)
        payload["prompt"] = prompt

        if "send_images" not in payload:
            payload["send_images"] = self.default_send_images

        effective_model = model or self.model
        if effective_model and self.use_override_settings:
            override = payload.get("override_settings")
            if override is None:
                override = {}
            elif not isinstance(override, dict):
                raise TypeError("override_settings must be a dict when provided")

            override = dict(override)
            override["sd_model_checkpoint"] = effective_model
            payload["override_settings"] = override
            payload.setdefault("override_settings_restore_afterwards", True)

        src_img = self._coerce_image_result(image)
        if src_img is None:
            return "/sdapi/v1/txt2img", payload

        payload["init_images"] = [_encode_b64_image_bytes(src_img.bytes())]
        payload.setdefault("denoising_strength", 0.75)
        payload.setdefault("include_init_images", False)

        return "/sdapi/v1/img2img", payload

    # ------------------------------------------------------------------
    # One-shot generation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Picture:
        """
        Unified one-shot generation.

        - `image is None` -> txt2img
        - `image provided` -> img2img

        Extra kwargs are passed directly to the A1111 payload
        (steps, width, height, seed, cfg_scale, sampler_name, negative_prompt, etc).
        """
        image_index = int(kwargs.pop("image_index", 0))
        endpoint, payload = self._build_generation_request(
            prompt,
            image=image,
            model=model,
            **kwargs,
        )

        with self._client() as client:
            res = client.post(self._url(endpoint), json=payload, auth=self.auth)

        if res.status_code != 200:
            raise RuntimeError(
                f"{endpoint} failed: HTTP {res.status_code} body={res.text[:500]!r}"
            )

        j = res.json()
        if not isinstance(j, dict):
            raise RuntimeError(f"Unexpected response: {type(j).__name__}")

        images = j.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError("generation returned no images")

        if image_index < 0 or image_index >= len(images):
            raise IndexError(
                f"Requested image_index={image_index}, but response has {len(images)} images"
            )

        raw_b64 = images[image_index]
        if not isinstance(raw_b64, str):
            raise RuntimeError(
                f"Unexpected image payload type: {type(raw_b64).__name__}"
            )

        img_bytes = _decode_b64_image_bytes(raw_b64)

        info_raw = j.get("info")
        info_parsed = _safe_json_loads(info_raw)

        mode = "img2img" if image is not None else "txt2img"
        metadata: Dict[str, Any] = {
            "provider": "stabdiff",
            "server": self.server,
            "model": (model or self.model),
            "endpoint": endpoint,
            "mode": mode,
            "parameters": j.get("parameters"),
            "info": info_parsed,
            "num_images": len(images),
            "image_index": image_index,
        }

        return Picture.from_bytes(
            img_bytes,
            raw=res,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Progress / preview polling
    # ------------------------------------------------------------------

    def _get_progress_sync(self, *, skip_current_image: bool = False) -> Dict[str, Any]:
        with self._client(timeout=self.progress_timeout) as client:
            res = client.get(
                self._url("/sdapi/v1/progress"),
                params={"skip_current_image": str(skip_current_image).lower()},
                auth=self.auth,
            )

        if res.status_code != 200:
            raise RuntimeError(f"progress failed: HTTP {res.status_code}")

        j = res.json()
        if not isinstance(j, dict):
            raise RuntimeError(f"Unexpected progress response: {type(j).__name__}")
        return j

    async def _get_progress_async(
        self,
        client: httpx.AsyncClient,
        *,
        skip_current_image: bool = False,
    ) -> Dict[str, Any]:
        res = await client.get(
            self._url("/sdapi/v1/progress"),
            params={"skip_current_image": str(skip_current_image).lower()},
            auth=self.auth,
        )

        if res.status_code != 200:
            raise RuntimeError(f"progress failed: HTTP {res.status_code}")

        j = res.json()
        if not isinstance(j, dict):
            raise RuntimeError(f"Unexpected progress response: {type(j).__name__}")
        return j

    def interrupt(self) -> None:
        with self._client(timeout=min(self.timeout or 30.0, 30.0)) as client:
            client.post(self._url("/sdapi/v1/interrupt"), auth=self.auth)

    async def ainterrupt(self) -> None:
        async with self._aclient(timeout=min(self.timeout or 30.0, 30.0)) as client:
            await client.post(self._url("/sdapi/v1/interrupt"), auth=self.auth)

    # ------------------------------------------------------------------
    # Provider-level streaming
    # ------------------------------------------------------------------

    async def astream_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Provider-level async stream for ImageMixin.imagestream(...).

        Emits:
          - {"type":"start", ...}
          - {"type":"progress", "progress":..., "eta_relative":..., ...}
             (optionally includes a preview Picture under "image")
          - {"type":"end", "image": Picture, ...}
          - {"type":"error", "message":"..."}
        """
        poll_interval = float(
            kwargs.pop("progress_poll_interval", self.progress_poll_interval)
        )
        include_previews = _coerce_bool(
            kwargs.pop("include_previews", self.include_previews)
        )
        skip_current_image = _coerce_bool(
            kwargs.pop("skip_current_image", (not include_previews))
        )
        interrupt_on_cancel = _coerce_bool(kwargs.pop("interrupt_on_cancel", True))

        actual_model = model or self.model
        mode = "img2img" if image is not None else "txt2img"

        yield {
            "type": "start",
            "provider": "stabdiff",
            "model": actual_model,
            "server": self.server,
            "mode": mode,
        }

        gen_task: asyncio.Task[Picture] = asyncio.create_task(
            asyncio.to_thread(
                self.generate_image,
                prompt,
                image,
                model,
                **kwargs,
            )
        )

        last_progress: Optional[float] = None
        last_eta: Optional[float] = None
        last_textinfo: Optional[str] = None
        last_preview_b64: Optional[str] = None

        progress_timeout = (
            self.progress_timeout if self.progress_timeout is not None else 10.0
        )

        try:
            async with self._aclient(timeout=progress_timeout) as client:
                while not gen_task.done():
                    try:
                        pj = await self._get_progress_async(
                            client,
                            skip_current_image=skip_current_image,
                        )

                        p = pj.get("progress")
                        eta = pj.get("eta_relative")
                        textinfo = pj.get("textinfo")

                        progress_changed = (
                            (isinstance(p, (int, float)) and p != last_progress)
                            or (isinstance(eta, (int, float)) and eta != last_eta)
                            or (isinstance(textinfo, str) and textinfo != last_textinfo)
                        )

                        preview_img: Optional[Picture] = None
                        if include_previews:
                            cur = pj.get("current_image")
                            if isinstance(cur, str) and cur and cur != last_preview_b64:
                                try:
                                    b = _decode_b64_image_bytes(cur)
                                    preview_img = Picture.from_bytes(
                                        b,
                                        metadata={
                                            "provider": "stabdiff",
                                            "server": self.server,
                                            "model": actual_model,
                                            "mode": mode,
                                            "preview": True,
                                            "progress": p,
                                            "eta_relative": eta,
                                            "textinfo": textinfo,
                                        },
                                    )
                                    last_preview_b64 = cur
                                except Exception:
                                    # Ignore malformed preview frames and continue polling
                                    pass

                        if progress_changed or preview_img is not None:
                            ev: Dict[str, Any] = {
                                "type": "progress",
                                "progress": p,
                                "eta_relative": eta,
                                "state": pj.get("state"),
                                "textinfo": textinfo,
                            }
                            if preview_img is not None:
                                ev["image"] = preview_img
                            yield ev

                        if isinstance(p, (int, float)):
                            last_progress = float(p)
                        if isinstance(eta, (int, float)):
                            last_eta = float(eta)
                        if isinstance(textinfo, str):
                            last_textinfo = textinfo

                    except Exception as e:
                        yield {
                            "type": "progress-error",
                            "message": str(e),
                        }

                    if not gen_task.done():
                        await asyncio.sleep(max(0.05, poll_interval))

            try:
                final_img = await gen_task
            except Exception as e:
                yield {"type": "error", "message": str(e)}
                return

            yield {
                "type": "end",
                "image": final_img,
                "content": final_img,  # harmless; ImageMixin normalizes to "image"
                "model": actual_model,
                "server": self.server,
                "mode": mode,
            }

        except asyncio.CancelledError:
            if interrupt_on_cancel:
                with suppress(Exception):
                    await self.ainterrupt()

            if not gen_task.done():
                gen_task.cancel()
                with suppress(Exception):
                    await gen_task

            raise
