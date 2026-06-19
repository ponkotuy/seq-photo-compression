from __future__ import annotations

import json
import struct
from pathlib import Path

from seq_photo_compression.errors import SpcError


MAGIC = b"SPCNEF1\0"
ARCHIVE_EXT = ".spcraw"


def read_archive(path: Path) -> tuple[dict, bytes, bytes]:
    if not path.is_file():
        raise SpcError(f"archive not found: {path}")
    with path.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise SpcError(f"not an SPC archive: {path}")
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        shell_len = int(header["chunks"]["shell_zstd_len"])
        diff_len = int(header["chunks"]["diff_zstd_len"])
        shell_zstd = f.read(shell_len)
        diff_zstd = f.read(diff_len)
        if len(shell_zstd) != shell_len or len(diff_zstd) != diff_len:
            raise SpcError("archive is truncated")
    return header, shell_zstd, diff_zstd


def write_archive(path: Path, header: dict, shell_zstd: bytes, diff_zstd: bytes, *, force: bool) -> None:
    if path.exists() and not force:
        raise SpcError(f"output exists, pass --force to overwrite: {path}")
    header = dict(header)
    header["chunks"] = {
        "shell_zstd_len": len(shell_zstd),
        "diff_zstd_len": len(diff_zstd),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(shell_zstd)
        f.write(diff_zstd)


def default_archive_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ARCHIVE_EXT)
