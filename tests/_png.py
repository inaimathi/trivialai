"""Test fixture: valid PNG bytes built with the stdlib only (no Pillow)."""

import struct
import zlib


def make_png(width: int = 1, height: int = 1, rgba=(255, 0, 0, 255)) -> bytes:
    """A valid RGBA PNG built with the stdlib only (no Pillow)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgba) * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
