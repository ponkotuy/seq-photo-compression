from __future__ import annotations

import json
import struct
from pathlib import Path

from seq_photo_compression.errors import SpcError


MAGIC = b"SPCNEF1\0"
ARCHIVE_EXT = ".spcraw"


def read_archive_chunks(path: Path) -> tuple[dict, dict[str, bytes]]:
    if not path.is_file():
        raise SpcError(f"archive not found: {path}")
    with path.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise SpcError(f"not an SPC archive: {path}")
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        chunk_lengths = header.get("chunks")
        if not isinstance(chunk_lengths, dict):
            raise SpcError("archive does not contain chunk metadata")

        chunk_order = header.get("chunk_order")
        if chunk_order is None:
            if "shell_zstd_len" in chunk_lengths and "diff_zstd_len" in chunk_lengths:
                chunk_order = ["shell_zstd", "diff_zstd"]
            else:
                raise SpcError("archive does not contain chunk order metadata")

        chunks: dict[str, bytes] = {}
        for chunk_name in chunk_order:
            if not isinstance(chunk_name, str):
                raise SpcError("invalid chunk order metadata")
            chunk_len = int(chunk_lengths[f"{chunk_name}_len"])
            chunk = f.read(chunk_len)
            if len(chunk) != chunk_len:
                raise SpcError("archive is truncated")
            chunks[chunk_name] = chunk

        if f.read(1):
            raise SpcError("archive has trailing data")
    return header, chunks


def read_archive(path: Path) -> tuple[dict, bytes, bytes]:
    header, chunks = read_archive_chunks(path)
    try:
        return header, chunks["shell_zstd"], chunks["diff_zstd"]
    except KeyError as exc:
        raise SpcError("archive is not a diff archive") from exc


def write_archive_chunks(path: Path, header: dict, chunks: dict[str, bytes], *, force: bool) -> None:
    if path.exists() and not force:
        raise SpcError(f"output exists, pass --force to overwrite: {path}")
    header = dict(header)
    header["chunk_order"] = list(chunks)
    header["chunks"] = {f"{name}_len": len(value) for name, value in chunks.items()}
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for chunk in chunks.values():
            f.write(chunk)


def write_archive(path: Path, header: dict, shell_zstd: bytes, diff_zstd: bytes, *, force: bool) -> None:
    write_archive_chunks(
        path,
        header,
        {
            "shell_zstd": shell_zstd,
            "diff_zstd": diff_zstd,
        },
        force=force,
    )


def default_archive_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ARCHIVE_EXT)
