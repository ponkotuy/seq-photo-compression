from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np

from seq_photo_compression.errors import SpcError
from seq_photo_compression.external import require_command, run_checked


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_raw_array(nef_path: Path) -> np.ndarray:
    require_command("unprocessed_raw")
    nef_path = nef_path.resolve()
    if not nef_path.is_file():
        raise SpcError(f"NEF not found: {nef_path}")

    with tempfile.TemporaryDirectory(prefix="spcraw-") as tmp:
        tmp_dir = Path(tmp)
        link_path = tmp_dir / "input.NEF"
        link_path.symlink_to(nef_path)
        run_checked(["unprocessed_raw", "-q", str(link_path)])
        pgm_path = tmp_dir / "input.NEF.pgm"
        return read_pgm_u16(pgm_path)


def _read_pnm_token(f: BinaryIO) -> bytes:
    token = bytearray()
    while True:
        c = f.read(1)
        if not c:
            raise SpcError("unexpected EOF in PGM header")
        if c == b"#":
            f.readline()
            continue
        if c.isspace():
            continue
        token.extend(c)
        break

    while True:
        c = f.read(1)
        if not c or c.isspace():
            break
        token.extend(c)
    return bytes(token)


def read_pgm_u16(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = _read_pnm_token(f)
        if magic != b"P5":
            raise SpcError(f"unsupported PGM magic: {magic!r}")
        width = int(_read_pnm_token(f))
        height = int(_read_pnm_token(f))
        maxval = int(_read_pnm_token(f))
        if maxval > 65535:
            raise SpcError(f"unsupported PGM max value: {maxval}")
        data = f.read()

    dtype = np.dtype(">u2") if maxval > 255 else np.dtype("u1")
    expected = width * height * dtype.itemsize
    if len(data) != expected:
        raise SpcError(f"PGM data size mismatch: expected {expected}, got {len(data)}")
    arr = np.frombuffer(data, dtype=dtype)
    return arr.astype(np.uint16, copy=True).reshape((height, width))


def raw_to_little_endian_bytes(raw: np.ndarray) -> bytes:
    return raw.astype("<u2", copy=False).tobytes(order="C")
