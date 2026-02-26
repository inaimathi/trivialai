# src/trivialai/image.py
from __future__ import annotations

import os
import tempfile
from asyncio import to_thread
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from PIL import Image as PILImage

from .bistream import BiStream


# ---------------------------------------------------------------------------
# ImageResult
# ---------------------------------------------------------------------------
def _looks_like_pil_image(obj: Any) -> bool:
    return isinstance(obj, PILImage.Image)


def _mime_to_ext(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None
    m = mime.lower().strip()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }.get(m)


def _mime_to_pil_format(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None
    m = mime.lower().strip()
    return {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/webp": "WEBP",
        "image/gif": "GIF",
        "image/bmp": "BMP",
        "image/tiff": "TIFF",
    }.get(m)


def _ext_to_pil_format(ext: Optional[str]) -> Optional[str]:
    if not ext:
        return None
    e = ext.lower().strip()
    if not e.startswith("."):
        e = f".{e}"
    return {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
        ".gif": "GIF",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }.get(e)


def _guess_ext_from_bytes(data: Optional[bytes]) -> Optional[str]:
    if not data:
        return None
    b = data
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if b.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    if b.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if b.startswith(b"BM"):
        return ".bmp"
    if b.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    return None


def _guess_media_type_from_bytes(data: Optional[bytes]) -> Optional[str]:
    ext = _guess_ext_from_bytes(data)
    if ext == ".png":
        return "image/png"
    if ext == ".jpg":
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".tiff":
        return "image/tiff"
    return None


class ImageResult:
    """
    Lazy image wrapper that can be backed by:
      - raw bytes
      - an on-disk file
      - a PIL image object

    The caller can request whichever representation they want:
      - .bytes()
      - .file(...)
      - .pil_image()

    Conversions are performed lazily and cached.
    """

    def __init__(
        self,
        *,
        data: Optional[bytes] = None,
        file_path: Optional[str | os.PathLike[str]] = None,
        pil: Any = None,
        media_type: Optional[str] = None,
        fmt: Optional[str] = None,
        raw: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._bytes: Optional[bytes] = bytes(data) if data is not None else None
        self._file_path: Optional[str] = (
            str(Path(file_path)) if file_path is not None else None
        )
        self._pil: Any = pil
        self.media_type = media_type
        self.format = fmt.upper() if isinstance(fmt, str) else fmt
        self.raw = raw
        self.metadata: Dict[str, Any] = dict(metadata or {})

        if self.media_type is None and self._bytes is not None:
            self.media_type = _guess_media_type_from_bytes(self._bytes)

        if self.format is None and self.media_type is not None:
            self.format = _mime_to_pil_format(self.media_type)

        if self._pil is None and self._bytes is None and self._file_path is None:
            raise ValueError("ImageResult requires one of: data, file_path, pil")

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: Optional[str] = None,
        fmt: Optional[str] = None,
        raw: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ImageResult":
        return cls(
            data=data,
            media_type=media_type,
            fmt=fmt,
            raw=raw,
            metadata=metadata,
        )

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        media_type: Optional[str] = None,
        fmt: Optional[str] = None,
        raw: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ImageResult":
        return cls(
            file_path=path,
            media_type=media_type,
            fmt=fmt,
            raw=raw,
            metadata=metadata,
        )

    @classmethod
    def from_pil(
        cls,
        img: Any,
        *,
        media_type: Optional[str] = None,
        fmt: Optional[str] = None,
        raw: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ImageResult":
        return cls(
            pil=img,
            media_type=media_type,
            fmt=fmt,
            raw=raw,
            metadata=metadata,
        )

    def _preferred_ext(self) -> str:
        ext = _mime_to_ext(self.media_type)
        if ext:
            return ext

        if self.format:
            fmt = self.format.upper()
            if fmt == "JPEG":
                return ".jpg"
            if fmt == "PNG":
                return ".png"
            if fmt == "WEBP":
                return ".webp"
            if fmt == "GIF":
                return ".gif"
            if fmt == "BMP":
                return ".bmp"
            if fmt == "TIFF":
                return ".tiff"

        ext = _guess_ext_from_bytes(self._bytes)
        if ext:
            return ext

        if self._file_path:
            p = Path(self._file_path)
            if p.suffix:
                return p.suffix.lower()

        return ".png"

    def _serialize_pil_to_bytes(self) -> bytes:
        if self._pil is None:
            raise RuntimeError("No PIL image available to serialize")

        img = self._pil
        fmt = self.format
        if not fmt:
            fmt = getattr(img, "format", None)
        if not fmt and self.media_type:
            fmt = _mime_to_pil_format(self.media_type)
        if not fmt and self._file_path:
            fmt = _ext_to_pil_format(Path(self._file_path).suffix)
        if not fmt:
            fmt = "PNG"

        buf = BytesIO()
        img.save(buf, format=fmt)
        out = buf.getvalue()

        self.format = fmt
        if not self.media_type:
            self.media_type = _guess_media_type_from_bytes(out) or self.media_type
        return out

    def bytes(self) -> bytes:
        if self._bytes is not None:
            return self._bytes

        if self._file_path is not None:
            with open(self._file_path, "rb") as f:
                self._bytes = f.read()
            if not self.media_type:
                self.media_type = _guess_media_type_from_bytes(self._bytes)
            return self._bytes

        if self._pil is not None:
            self._bytes = self._serialize_pil_to_bytes()
            return self._bytes

        raise RuntimeError("ImageResult has no source representation")

    def pil_image(self):
        if self._pil is not None:
            return self._pil

        if self._bytes is not None:
            with BytesIO(self._bytes) as bio:
                img = PILImage.open(bio)
                img.load()
                self._pil = img.copy()
            if not self.format:
                self.format = getattr(self._pil, "format", None) or self.format
            if not self.media_type:
                self.media_type = _guess_media_type_from_bytes(self._bytes)
            return self._pil

        if self._file_path is not None:
            with PILImage.open(self._file_path) as img:
                img.load()
                self._pil = img.copy()
                if not self.format:
                    self.format = getattr(img, "format", None) or self.format
            return self._pil

        raise RuntimeError("ImageResult has no source representation")

    def file(
        self,
        path: Optional[str | os.PathLike[str]] = None,
        *,
        directory: Optional[str | os.PathLike[str]] = None,
        prefix: str = "trivialai-img-",
        suffix: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        if path is None and self._file_path is not None:
            return self._file_path

        chosen_suffix = suffix or self._preferred_ext()

        if path is None:
            fd, tmp = tempfile.mkstemp(
                prefix=prefix,
                suffix=chosen_suffix,
                dir=(str(directory) if directory is not None else None),
            )
            os.close(fd)
            out_path = tmp
        else:
            out_path = str(Path(path))
            if os.path.exists(out_path) and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {out_path!r} "
                    "(pass overwrite=True to allow)"
                )

            parent = str(Path(out_path).parent)
            if parent and parent != ".":
                os.makedirs(parent, exist_ok=True)

        if self._bytes is not None:
            with open(out_path, "wb") as f:
                f.write(self._bytes)
        elif self._file_path is not None:
            with open(self._file_path, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        elif self._pil is not None:
            fmt = self.format or _ext_to_pil_format(Path(out_path).suffix) or "PNG"
            self._pil.save(out_path, format=fmt)
            self.format = fmt
        else:
            raise RuntimeError("ImageResult has no source representation")

        self._file_path = out_path
        if not self.media_type and self._bytes is not None:
            self.media_type = _guess_media_type_from_bytes(self._bytes)
        return out_path

    def exists(self) -> bool:
        return self._file_path is not None and os.path.exists(self._file_path)

    @property
    def path(self) -> Optional[str]:
        return self._file_path

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._file_path:
            parts.append(f"path={self._file_path!r}")
        if self._bytes is not None:
            parts.append(f"bytes={len(self._bytes)}")
        if self._pil is not None:
            size = getattr(self._pil, "size", None)
            mode = getattr(self._pil, "mode", None)
            if size:
                parts.append(f"pil={size}")
            if mode:
                parts.append(f"mode={mode}")
        if self.media_type:
            parts.append(f"media_type={self.media_type!r}")
        if self.format:
            parts.append(f"format={self.format!r}")
        return f"ImageResult({', '.join(parts)})"


# ---------------------------------------------------------------------------
# ImageMixin
# ---------------------------------------------------------------------------


class ImageMixin:
    """
    Mixin surface for image-generation models.

    Unified interface:
      - imagen(prompt, image=None, model=None, **kwargs) -> ImageResult
      - aimagen(prompt, image=None, model=None, **kwargs) -> ImageResult
      - imagestream(prompt, image=None, model=None, **kwargs) -> BiStream[dict]

    Providers may interpret `image` as:
      - img2img init image
      - edit/inpaint source image
      - conditioning image
      - ignored (for text-only image models)
    """

    def _coerce_image_result(self, obj: Any) -> Optional[ImageResult]:
        if obj is None:
            return None

        if isinstance(obj, ImageResult):
            return obj

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return ImageResult.from_bytes(bytes(obj))

        if isinstance(obj, (str, os.PathLike)):
            return ImageResult.from_file(str(obj))

        if _looks_like_pil_image(obj):
            return ImageResult.from_pil(obj)

        return None

    def _event_image_result(self, ev: Dict[str, Any]) -> Optional[ImageResult]:
        if "image" in ev:
            return self._coerce_image_result(ev.get("image"))

        for k in ("result", "pil_image", "pil", "bytes", "data", "file", "path"):
            if k in ev:
                return self._coerce_image_result(ev.get(k))

        if "content" in ev:
            content = ev.get("content")
            if isinstance(content, (bytes, bytearray, memoryview)):
                return self._coerce_image_result(content)
            if _looks_like_pil_image(content):
                return self._coerce_image_result(content)
            if isinstance(content, (str, os.PathLike)) and os.path.exists(str(content)):
                return self._coerce_image_result(content)

        return None

    def _require_image_result(self, obj: Any) -> ImageResult:
        out = self._coerce_image_result(obj)
        if out is None:
            raise TypeError(
                "Expected image-like result (ImageResult/bytes/file path/PIL image), "
                f"got {type(obj).__name__}: {obj!r}"
            )
        return out

    def generate_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ImageResult | bytes | str | os.PathLike[str] | Any:
        raise NotImplementedError

    async def agenerate_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ImageResult:
        raw = await to_thread(self.generate_image, prompt, image, model, **kwargs)
        return self._require_image_result(raw)

    async def astream_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        actual_model = model or getattr(self, "model", None)

        yield {
            "type": "start",
            "provider": self.__class__.__name__.lower(),
            "model": actual_model,
        }

        img = await self.agenerate_image(prompt, image=image, model=model, **kwargs)

        yield {
            "type": "end",
            "image": img,
        }

    def imagen(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ImageResult:
        raw = self.generate_image(prompt, image=image, model=model, **kwargs)
        return self._require_image_result(raw)

    async def aimagen(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ImageResult:
        last_preview: Optional[ImageResult] = None
        last_error: Optional[str] = None

        async for ev in self.imagestream(prompt, image=image, model=model, **kwargs):
            if not isinstance(ev, dict):
                continue
            t = ev.get("type")
            if t == "error":
                last_error = str(ev.get("message") or ev.get("error") or "image-error")
            img = ev.get("image")
            if isinstance(img, ImageResult):
                last_preview = img
                if t == "end":
                    return img

        if last_error:
            raise RuntimeError(last_error)
        if last_preview is not None:
            return last_preview

        raise RuntimeError("imagestream ended without producing an image")

    def imagestream(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BiStream[Dict[str, Any]]:
        async def _wrapped() -> AsyncIterator[Dict[str, Any]]:
            last_image: Optional[ImageResult] = None

            async for ev in self.astream_image(
                prompt, image=image, model=model, **kwargs
            ):
                if not isinstance(ev, dict):
                    img = self._coerce_image_result(ev)
                    if img is not None:
                        last_image = img
                        yield {"type": "delta", "image": img}
                    else:
                        yield {"type": "delta", "value": ev}
                    continue

                new_ev = dict(ev)
                img = self._event_image_result(new_ev)
                if img is not None:
                    new_ev["image"] = img
                    last_image = img

                if new_ev.get("type") == "end":
                    if "image" not in new_ev or not isinstance(
                        new_ev["image"], ImageResult
                    ):
                        if last_image is None:
                            raise RuntimeError(
                                "Provider emitted end event without an image payload"
                            )
                        new_ev["image"] = last_image

                yield new_ev

        return BiStream(_wrapped())
