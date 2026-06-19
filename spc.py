#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from spc_motion import MotionResidualError, encode_residual_pam, restore_from_residual_pam


MAGIC = b"SPCNEF1\0"
ARCHIVE_EXT = ".spcraw"

TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_STRIP_OFFSETS = 273
TAG_STRIP_BYTE_COUNTS = 279
TAG_SUB_IFDS = 330

TIFF_COMPRESSION_NONE = 1
TIFF_COMPRESSION_NIKON_NEF = 34713

TYPE_SHORT = 3
TYPE_LONG = 4
TYPE_SIZES = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    7: 1,  # UNDEFINED
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
}


class SpcError(RuntimeError):
    pass


@dataclass(frozen=True)
class TiffEntry:
    tag: int
    type_id: int
    count: int
    value_offset_pos: int
    data_offset: int


@dataclass(frozen=True)
class TiffIfd:
    offset: int
    entries: dict[int, TiffEntry]


@dataclass(frozen=True)
class RawStripInfo:
    strip_offset: int
    strip_byte_count: int
    compression: int
    bits_per_sample: int | None
    compression_entry_pos: int
    strip_offset_entry_pos: int
    strip_byte_count_entry_pos: int
    bits_per_sample_entry_pos: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SpcError(f"required command not found: {command}")


def run_checked(args: list[str], *, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            args,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise SpcError(f"command failed: {' '.join(args)}\n{stderr}") from exc
    return result.stdout


def zstd_compress(data: bytes, level: int) -> bytes:
    require_command("zstd")
    return run_checked(["zstd", "-q", f"-{level}", "-T0", "-c"], input_bytes=data)


def zstd_decompress(data: bytes) -> bytes:
    require_command("zstd")
    return run_checked(["zstd", "-q", "-d", "-c"], input_bytes=data)


def encode_jxl_residual(
    base_raw: np.ndarray,
    target_raw: np.ndarray,
    *,
    motion_mode: str,
    effort: int,
) -> tuple[bytes, dict]:
    require_command("cjxl")

    with tempfile.TemporaryDirectory(prefix="spc-jxl-") as tmp:
        tmp_dir = Path(tmp)
        residual_pam = tmp_dir / "residual.pam"
        residual_jxl = tmp_dir / "residual.jxl"
        try:
            stats = encode_residual_pam(
                base_raw,
                target_raw,
                motion_mode=motion_mode,
                output=residual_pam,
            )
        except MotionResidualError as exc:
            raise SpcError(str(exc)) from exc
        run_checked(
            [
                "cjxl",
                str(residual_pam),
                str(residual_jxl),
                "--distance=0",
                "--modular=1",
                f"--effort={effort}",
                "--quiet",
            ]
        )
        payload = residual_jxl.read_bytes()

    metadata = {
        "compression": "jxl_modular",
        "source": "rggb4_residual_u16_pam",
        "motion_mode": motion_mode,
        "motion_status": stats.status,
        "motion_score": stats.score,
        "motion_matrix": stats.matrix,
        "residual_min": stats.residual_min,
        "residual_max": stats.residual_max,
        "residual_offset": stats.offset,
        "jxl_effort": effort,
    }
    return payload, metadata


def decode_jxl_residual(
    header: dict,
    base_raw: np.ndarray,
    residual_jxl: bytes,
) -> np.ndarray:
    require_command("djxl")
    height = int(header["raw"]["height"])
    width = int(header["raw"]["width"])
    diff_info = header["diff"]

    with tempfile.TemporaryDirectory(prefix="spc-jxl-") as tmp:
        tmp_dir = Path(tmp)
        residual_jxl_path = tmp_dir / "residual.jxl"
        residual_pam_path = tmp_dir / "residual.pam"
        residual_jxl_path.write_bytes(residual_jxl)
        run_checked(["djxl", str(residual_jxl_path), str(residual_pam_path), "--quiet"])
        try:
            restored = restore_from_residual_pam(
                base_raw,
                residual_pam_path,
                motion_mode=str(diff_info["motion_mode"]),
                matrix=str(diff_info["motion_matrix"]),
            )
        except MotionResidualError as exc:
            raise SpcError(str(exc)) from exc

    if restored.shape != (height, width):
        raise SpcError(f"restored RAW shape mismatch: expected {(height, width)}, got {restored.shape}")
    return restored


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


class TiffParser:
    def __init__(self, data: bytes | bytearray):
        self.data = data
        if data[:2] == b"II":
            self.endian = "<"
        elif data[:2] == b"MM":
            self.endian = ">"
        else:
            raise SpcError("not a TIFF/NEF file")
        if self.u16(2) != 42:
            raise SpcError("unsupported TIFF magic")
        self.first_ifd = self.u32(4)

    def u16(self, offset: int) -> int:
        return struct.unpack_from(self.endian + "H", self.data, offset)[0]

    def u32(self, offset: int) -> int:
        return struct.unpack_from(self.endian + "I", self.data, offset)[0]

    def parse_ifd(self, offset: int) -> TiffIfd:
        if offset <= 0 or offset + 2 > len(self.data):
            raise SpcError(f"invalid IFD offset: {offset}")
        count = self.u16(offset)
        entries: dict[int, TiffEntry] = {}
        pos = offset + 2
        for _ in range(count):
            tag = self.u16(pos)
            type_id = self.u16(pos + 2)
            value_count = self.u32(pos + 4)
            value_offset_pos = pos + 8
            size = TYPE_SIZES.get(type_id)
            if size is None:
                data_offset = self.u32(value_offset_pos)
            elif size * value_count <= 4:
                data_offset = value_offset_pos
            else:
                data_offset = self.u32(value_offset_pos)
            entries[tag] = TiffEntry(tag, type_id, value_count, value_offset_pos, data_offset)
            pos += 12
        return TiffIfd(offset, entries)

    def next_ifd_offset(self, ifd: TiffIfd) -> int:
        count = self.u16(ifd.offset)
        return self.u32(ifd.offset + 2 + count * 12)

    def entry_values(self, entry: TiffEntry) -> list[int]:
        if entry.type_id == TYPE_SHORT:
            fmt = self.endian + ("H" * entry.count)
            size = 2 * entry.count
        elif entry.type_id == TYPE_LONG:
            fmt = self.endian + ("I" * entry.count)
            size = 4 * entry.count
        else:
            raise SpcError(f"unsupported tag type for numeric values: {entry.type_id}")
        if entry.data_offset + size > len(self.data):
            raise SpcError("TIFF tag points outside file")
        return list(struct.unpack_from(fmt, self.data, entry.data_offset))

    def all_ifds(self) -> list[TiffIfd]:
        result: list[TiffIfd] = []
        seen: set[int] = set()

        def visit(offset: int) -> None:
            while offset and offset not in seen:
                seen.add(offset)
                ifd = self.parse_ifd(offset)
                result.append(ifd)
                subifd = ifd.entries.get(TAG_SUB_IFDS)
                if subifd is not None:
                    for sub_offset in self.entry_values(subifd):
                        visit(sub_offset)
                offset = self.next_ifd_offset(ifd)

        visit(self.first_ifd)
        return result

    def find_raw_strip(self) -> RawStripInfo:
        best: tuple[int, TiffIfd] | None = None
        for ifd in self.all_ifds():
            offsets = ifd.entries.get(TAG_STRIP_OFFSETS)
            counts = ifd.entries.get(TAG_STRIP_BYTE_COUNTS)
            compression = ifd.entries.get(TAG_COMPRESSION)
            if offsets is None or counts is None or compression is None:
                continue
            byte_counts = self.entry_values(counts)
            if len(byte_counts) != 1:
                continue
            if best is None or byte_counts[0] > best[0]:
                best = (byte_counts[0], ifd)

        if best is None:
            raise SpcError("could not find raw strip in NEF")
        ifd = best[1]
        strip_offsets = self.entry_values(ifd.entries[TAG_STRIP_OFFSETS])
        strip_counts = self.entry_values(ifd.entries[TAG_STRIP_BYTE_COUNTS])
        if len(strip_offsets) != 1 or len(strip_counts) != 1:
            raise SpcError("only single-strip NEF files are supported in phase 1")
        compression_value = self.entry_values(ifd.entries[TAG_COMPRESSION])[0]
        bits_entry = ifd.entries.get(TAG_BITS_PER_SAMPLE)
        bits_per_sample = self.entry_values(bits_entry)[0] if bits_entry is not None else None
        return RawStripInfo(
            strip_offset=strip_offsets[0],
            strip_byte_count=strip_counts[0],
            compression=compression_value,
            bits_per_sample=bits_per_sample,
            compression_entry_pos=ifd.entries[TAG_COMPRESSION].value_offset_pos,
            strip_offset_entry_pos=ifd.entries[TAG_STRIP_OFFSETS].value_offset_pos,
            strip_byte_count_entry_pos=ifd.entries[TAG_STRIP_BYTE_COUNTS].value_offset_pos,
            bits_per_sample_entry_pos=bits_entry.value_offset_pos if bits_entry is not None else None,
        )


def make_zeroed_shell(target_nef: Path) -> tuple[bytes, RawStripInfo]:
    if not target_nef.is_file():
        raise SpcError(f"target NEF not found: {target_nef}")
    data = bytearray(target_nef.read_bytes())
    parser = TiffParser(data)
    raw = parser.find_raw_strip()
    end = raw.strip_offset + raw.strip_byte_count
    if raw.strip_offset < 0 or end > len(data):
        raise SpcError("raw strip points outside target NEF")
    data[raw.strip_offset:end] = b"\0" * raw.strip_byte_count
    return bytes(data), raw


NIKON_LOSSLESS_14_TREE = (
    0,
    1,
    4,
    2,
    2,
    3,
    1,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    6,
    8,
    5,
    9,
    4,
    10,
    3,
    11,
    12,
    2,
    0,
    1,
    13,
    14,
)


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bitbuf = 0
        self.vbits = 0

    def get(self, nbits: int) -> int:
        while self.vbits < nbits:
            if self.byte_pos >= len(self.data):
                raise SpcError("unexpected EOF in Nikon compressed RAW")
            self.bitbuf = (self.bitbuf << 8) | self.data[self.byte_pos]
            self.byte_pos += 1
            self.vbits += 8
        shift = self.vbits - nbits
        value = (self.bitbuf >> shift) & ((1 << nbits) - 1)
        self.vbits -= nbits
        self.bitbuf &= (1 << self.vbits) - 1 if self.vbits else 0
        return value


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.bitbuf = 0
        self.vbits = 0

    def put(self, value: int, nbits: int) -> None:
        if nbits == 0:
            return
        self.bitbuf = (self.bitbuf << nbits) | (value & ((1 << nbits) - 1))
        self.vbits += nbits
        while self.vbits >= 8:
            shift = self.vbits - 8
            self.out.append((self.bitbuf >> shift) & 0xFF)
            self.vbits -= 8
            self.bitbuf &= (1 << self.vbits) - 1 if self.vbits else 0

    def finish(self, *, padding_bits: int = 0, padding_value: int = 0) -> bytes:
        if self.vbits:
            needed = 8 - self.vbits
            if padding_bits != needed:
                padding_value = 0
            self.out.append(((self.bitbuf << needed) | (padding_value & ((1 << needed) - 1))) & 0xFF)
            self.bitbuf = 0
            self.vbits = 0
        return bytes(self.out)


def build_huffman_codes(tree: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    symbol_pos = 16
    for bit_length, count in enumerate(tree[:16], start=1):
        for _ in range(count):
            symbol = tree[symbol_pos]
            symbol_pos += 1
            codes[symbol] = (code, bit_length)
            code += 1
        code <<= 1
    return codes


def build_huffman_decoder(tree: tuple[int, ...]) -> dict[tuple[int, int], int]:
    return {(length, code): symbol for symbol, (code, length) in build_huffman_codes(tree).items()}


NIKON_LOSSLESS_14_CODES = build_huffman_codes(NIKON_LOSSLESS_14_TREE)
NIKON_LOSSLESS_14_DECODER = build_huffman_decoder(NIKON_LOSSLESS_14_TREE)
NIKON_LOSSLESS_14_MAX_CODE_BITS = max(length for _code, length in NIKON_LOSSLESS_14_CODES.values())


def read_huffman_symbol(reader: BitReader) -> int:
    code = 0
    for length in range(1, NIKON_LOSSLESS_14_MAX_CODE_BITS + 1):
        code = (code << 1) | reader.get(1)
        symbol = NIKON_LOSSLESS_14_DECODER.get((length, code))
        if symbol is not None:
            return symbol
    raise SpcError("invalid Nikon lossless Huffman code")


def decode_nikon_lossless_14_diff(reader: BitReader) -> int:
    symbol = read_huffman_symbol(reader)
    length = symbol & 15
    shift = symbol >> 4
    if shift:
        raise SpcError("unsupported shifted Nikon lossless symbol")
    if length == 0:
        return 0
    diff_bits = reader.get(length)
    if diff_bits & (1 << (length - 1)):
        return diff_bits
    return diff_bits - ((1 << length) - 1)


def get_raw_strip_bytes(nef_path: Path, raw_info: RawStripInfo) -> bytes:
    with nef_path.open("rb") as f:
        f.seek(raw_info.strip_offset)
        return f.read(raw_info.strip_byte_count)


def derive_nikon_lossless_14_restore_info(
    target_nef: Path,
    raw_info: RawStripInfo,
    target_raw: np.ndarray,
) -> dict | None:
    if raw_info.compression != TIFF_COMPRESSION_NIKON_NEF or raw_info.bits_per_sample != 14:
        return None

    strip = get_raw_strip_bytes(target_nef, raw_info)
    reader = BitReader(strip)
    raw_view = target_raw
    height, width = raw_view.shape
    vpred: list[list[int | None]] = [[None, None], [None, None]]
    initial_vpred: list[list[int | None]] = [[None, None], [None, None]]
    hpred = [0, 0]

    for row in range(height):
        parity = row & 1
        for col in range(width):
            diff = decode_nikon_lossless_14_diff(reader)
            expected = int(raw_view[row, col])
            if col < 2:
                current = vpred[parity][col]
                if current is None:
                    initial = expected - diff
                    if initial < 0 or initial > 0x3FFF:
                        raise SpcError("invalid Nikon lossless initial predictor")
                    initial_vpred[parity][col] = initial
                    vpred[parity][col] = expected
                    hpred[col] = expected
                    continue
                value = current + diff
                vpred[parity][col] = value
                hpred[col] = value
            else:
                value = hpred[col & 1] + diff
                hpred[col & 1] = value
            if value != expected:
                return None

    if any(item is None for row_values in initial_vpred for item in row_values):
        return None
    return {
        "codec": "nikon_lossless_14",
        "compression_tag": TIFF_COMPRESSION_NIKON_NEF,
        "bits_per_sample": 14,
        "initial_vpred": initial_vpred,
        "padding_bits": reader.vbits,
        "padding_value": reader.bitbuf,
        "trailing_bytes": strip[reader.byte_pos :].hex(),
    }


def encode_nikon_lossless_14(
    raw: np.ndarray,
    initial_vpred: list[list[int]],
    *,
    padding_bits: int = 0,
    padding_value: int = 0,
    trailing_bytes: bytes = b"",
) -> bytes:
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16, copy=False)
    codes = NIKON_LOSSLESS_14_CODES
    writer = BitWriter()
    vpred = [
        [int(initial_vpred[0][0]), int(initial_vpred[0][1])],
        [int(initial_vpred[1][0]), int(initial_vpred[1][1])],
    ]
    hpred = [0, 0]
    height, width = raw.shape

    for row in range(height):
        parity = row & 1
        for col in range(width):
            value = int(raw[row, col])
            if value < 0 or value > 0x3FFF:
                raise SpcError("Nikon lossless 14-bit output requires RAW values in 0..16383")
            if col < 2:
                pred = vpred[parity][col]
                diff = value - pred
                vpred[parity][col] = value
                hpred[col] = value
            else:
                pred = hpred[col & 1]
                diff = value - pred
                hpred[col & 1] = value

            length = abs(diff).bit_length()
            if length > 14:
                raise SpcError("Nikon lossless diff does not fit 14 bits")
            code, code_bits = codes[length]
            writer.put(code, code_bits)
            if length:
                if diff < 0:
                    diff_bits = diff + (1 << length) - 1
                else:
                    diff_bits = diff
                writer.put(diff_bits, length)
    return writer.finish(padding_bits=padding_bits, padding_value=padding_value) + trailing_bytes


def patch_short_inline(data: bytearray, endian: str, value_offset_pos: int, value: int) -> None:
    struct.pack_into(endian + "H", data, value_offset_pos, value)
    data[value_offset_pos + 2 : value_offset_pos + 4] = b"\0\0"


def patch_long_inline(data: bytearray, endian: str, value_offset_pos: int, value: int) -> None:
    struct.pack_into(endian + "I", data, value_offset_pos, value)


def patch_shell_for_uncompressed_raw(shell: bytes, raw_info: dict, raw_bytes: bytes) -> bytes:
    data = bytearray(shell)
    parser = TiffParser(data)
    append_offset = len(data)
    data.extend(raw_bytes)

    compression_pos = int(raw_info["compression_entry_pos"])
    strip_offset_pos = int(raw_info["strip_offset_entry_pos"])
    strip_byte_count_pos = int(raw_info["strip_byte_count_entry_pos"])

    patch_short_inline(data, parser.endian, compression_pos, TIFF_COMPRESSION_NONE)
    patch_long_inline(data, parser.endian, strip_offset_pos, append_offset)
    patch_long_inline(data, parser.endian, strip_byte_count_pos, len(raw_bytes))
    return bytes(data)


def patch_shell_for_nikon_compressed_raw(shell: bytes, target_shell_info: dict, raw_info: dict, compressed_raw: bytes) -> bytes:
    data = bytearray(shell)
    parser = TiffParser(data)
    original_offset = int(target_shell_info["zeroed_raw_strip_offset"])
    original_count = int(target_shell_info["zeroed_raw_strip_byte_count"])
    if len(compressed_raw) <= original_count:
        strip_offset = original_offset
        data[strip_offset : strip_offset + len(compressed_raw)] = compressed_raw
    else:
        strip_offset = len(data)
        data.extend(compressed_raw)

    compression_pos = int(raw_info["compression_entry_pos"])
    strip_offset_pos = int(raw_info["strip_offset_entry_pos"])
    strip_byte_count_pos = int(raw_info["strip_byte_count_entry_pos"])

    patch_short_inline(data, parser.endian, compression_pos, TIFF_COMPRESSION_NIKON_NEF)
    patch_long_inline(data, parser.endian, strip_offset_pos, strip_offset)
    patch_long_inline(data, parser.endian, strip_byte_count_pos, len(compressed_raw))
    return bytes(data)


def build_diff(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    if base.shape != target.shape:
        raise SpcError(f"RAW shape mismatch: base={base.shape}, target={target.shape}")
    diff32 = target.astype(np.int32) - base.astype(np.int32)
    if int(diff32.min()) < -32768 or int(diff32.max()) > 32767:
        raise SpcError("RAW difference does not fit int16")
    return diff32.astype("<i2", copy=False)


def restore_raw_from_diff(base: np.ndarray, diff: np.ndarray) -> np.ndarray:
    restored = base.astype(np.int32) + diff.astype(np.int32)
    if int(restored.min()) < 0 or int(restored.max()) > 65535:
        raise SpcError("restored RAW value outside uint16 range")
    return restored.astype(np.uint16)


def raw_to_little_endian_bytes(raw: np.ndarray) -> bytes:
    return raw.astype("<u2", copy=False).tobytes(order="C")


def load_diff(header: dict, diff_zstd: bytes) -> np.ndarray:
    raw_diff = zstd_decompress(diff_zstd)
    height = int(header["raw"]["height"])
    width = int(header["raw"]["width"])
    expected = height * width * np.dtype("<i2").itemsize
    if len(raw_diff) != expected:
        raise SpcError(f"diff size mismatch: expected {expected}, got {len(raw_diff)}")
    return np.frombuffer(raw_diff, dtype="<i2").reshape((height, width))


def restore_raw_from_archive_payload(header: dict, base_raw: np.ndarray, diff_payload: bytes) -> np.ndarray:
    compression = header["diff"]["compression"]
    if compression == "zstd":
        diff = load_diff(header, diff_payload)
        return restore_raw_from_diff(base_raw, diff)
    if compression == "jxl_modular":
        return decode_jxl_residual(header, base_raw, diff_payload)
    raise SpcError(f"unsupported diff compression: {compression}")


def default_archive_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ARCHIVE_EXT)


def cmd_encode(args: argparse.Namespace) -> None:
    keyframe = Path(args.keyframe)
    target = Path(args.target)
    output = Path(args.output) if args.output else default_archive_path(target)

    base_raw = extract_raw_array(keyframe)
    target_raw = extract_raw_array(target)

    shell, raw_info = make_zeroed_shell(target)
    try:
        nikon_restore_info = derive_nikon_lossless_14_restore_info(target, raw_info, target_raw)
    except SpcError:
        nikon_restore_info = None
    shell_zstd = zstd_compress(shell, args.zstd_level)
    if args.diff_codec == "zstd":
        diff = build_diff(base_raw, target_raw)
        diff_payload = zstd_compress(diff.tobytes(order="C"), args.zstd_level)
        diff_header = {
            "compression": "zstd",
            "min": int(diff.min()),
            "max": int(diff.max()),
        }
    elif args.diff_codec == "jxl":
        diff_payload, diff_header = encode_jxl_residual(
            base_raw,
            target_raw,
            motion_mode=args.motion_mode,
            effort=args.jxl_effort,
        )
    else:
        raise SpcError(f"unsupported diff codec: {args.diff_codec}")

    header = {
        "version": 1,
        "keyframe": {
            "path": str(keyframe),
            "sha256": sha256_file(keyframe),
            "size": keyframe.stat().st_size,
        },
        "target": {
            "path": str(target),
            "sha256": sha256_file(target),
            "size": target.stat().st_size,
        },
        "raw": {
            "width": int(target_raw.shape[1]),
            "height": int(target_raw.shape[0]),
            "dtype": "uint16",
            "diff_dtype": "int16",
        },
        "target_shell": {
            "compression": "zstd",
            "original_raw_compression_tag": raw_info.compression,
            "original_bits_per_sample": raw_info.bits_per_sample,
            "raw_restore": nikon_restore_info,
            "zeroed_raw_strip_offset": raw_info.strip_offset,
            "zeroed_raw_strip_byte_count": raw_info.strip_byte_count,
            "tiff_patch": {
                "compression_entry_pos": raw_info.compression_entry_pos,
                "strip_offset_entry_pos": raw_info.strip_offset_entry_pos,
                "strip_byte_count_entry_pos": raw_info.strip_byte_count_entry_pos,
                "bits_per_sample_entry_pos": raw_info.bits_per_sample_entry_pos,
            },
        },
        "diff": diff_header,
    }
    write_archive(output, header, shell_zstd, diff_payload, force=args.force)

    archive_size = output.stat().st_size
    target_size = target.stat().st_size
    ratio = archive_size / target_size
    print(f"archive: {output}")
    print(f"target NEF size: {target_size:,} bytes")
    print(f"archive size: {archive_size:,} bytes ({ratio:.2%} of target)")
    print(f"raw shape: {target_raw.shape[1]}x{target_raw.shape[0]}")
    if args.diff_codec == "zstd":
        print(f"diff codec: zstd")
        print(f"diff range: {int(diff_header['min'])}..{int(diff_header['max'])}")
    else:
        print(f"diff codec: jxl_modular")
        print(f"motion mode: {diff_header['motion_mode']}")
        print(f"motion status: {diff_header['motion_status']}")
        print(f"motion score: {diff_header['motion_score']:.6f}")
        print(f"residual range: {diff_header['residual_min']}..{diff_header['residual_max']}")
        print(f"jxl effort: {diff_header['jxl_effort']}")
    if nikon_restore_info is not None:
        print("restore RAW codec: nikon_lossless_14")
    else:
        print("restore RAW codec: uncompressed")


def cmd_verify(args: argparse.Namespace) -> None:
    keyframe = Path(args.keyframe)
    target = Path(args.target)
    archive = Path(args.archive)

    header, _shell_zstd, diff_payload = read_archive(archive)
    if sha256_file(keyframe) != header["keyframe"]["sha256"]:
        raise SpcError("keyframe SHA-256 does not match archive metadata")

    base_raw = extract_raw_array(keyframe)
    target_raw = extract_raw_array(target)
    restored = restore_raw_from_archive_payload(header, base_raw, diff_payload)
    equal = np.array_equal(restored, target_raw)
    differing = 0 if equal else int(np.count_nonzero(restored != target_raw))
    print(f"raw match: {equal}")
    print(f"differing pixels: {differing}")
    if not equal:
        raise SpcError("restored RAW does not match target RAW")


def cmd_restore(args: argparse.Namespace) -> None:
    keyframe = Path(args.keyframe)
    archive = Path(args.archive)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SpcError(f"output exists, pass --force to overwrite: {output}")

    header, shell_zstd, diff_payload = read_archive(archive)
    if sha256_file(keyframe) != header["keyframe"]["sha256"]:
        raise SpcError("keyframe SHA-256 does not match archive metadata")

    base_raw = extract_raw_array(keyframe)
    restored_raw = restore_raw_from_archive_payload(header, base_raw, diff_payload)

    shell = zstd_decompress(shell_zstd)
    raw_restore_info = header["target_shell"].get("raw_restore")
    if args.raw_output in ("auto", "nikon") and raw_restore_info is not None:
        if raw_restore_info.get("codec") != "nikon_lossless_14":
            raise SpcError(f"unsupported restore RAW codec: {raw_restore_info.get('codec')}")
        compressed_raw = encode_nikon_lossless_14(
            restored_raw,
            raw_restore_info["initial_vpred"],
            padding_bits=int(raw_restore_info.get("padding_bits", 0)),
            padding_value=int(raw_restore_info.get("padding_value", 0)),
            trailing_bytes=bytes.fromhex(str(raw_restore_info.get("trailing_bytes", ""))),
        )
        restored_nef = patch_shell_for_nikon_compressed_raw(
            shell,
            header["target_shell"],
            header["target_shell"]["tiff_patch"],
            compressed_raw,
        )
        restored_codec = "nikon_lossless_14"
    elif args.raw_output == "nikon":
        raise SpcError("archive does not contain Nikon lossless restore metadata; re-run encode with this version")
    else:
        raw_bytes = raw_to_little_endian_bytes(restored_raw)
        restored_nef = patch_shell_for_uncompressed_raw(shell, header["target_shell"]["tiff_patch"], raw_bytes)
        restored_codec = "uncompressed"
    output.write_bytes(restored_nef)
    print(f"restored NEF-like file: {output}")
    print(f"output size: {output.stat().st_size:,} bytes")
    print(f"restore RAW codec: {restored_codec}")
    xmp_sidecar = Path(str(output) + ".xmp")
    if xmp_sidecar.exists():
        print(
            f"warning: existing Darktable sidecar may override raw settings: {xmp_sidecar}",
            file=sys.stderr,
        )


def cmd_info(args: argparse.Namespace) -> None:
    header, _shell_zstd, _diff_zstd = read_archive(Path(args.archive))
    print(json.dumps(header, indent=2, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential NEF compression experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser("encode", help="create an archive for the second NEF using a keyframe NEF")
    encode.add_argument("keyframe", help="first NEF kept as keyframe")
    encode.add_argument("target", help="second NEF to encode as custom format")
    encode.add_argument("-o", "--output", help=f"output archive path, default: TARGET.NEF{ARCHIVE_EXT}")
    encode.add_argument("--diff-codec", choices=("zstd", "jxl"), default="jxl")
    encode.add_argument("--motion-mode", choices=("none", "translation", "ecc_affine"), default="ecc_affine")
    encode.add_argument("--jxl-effort", type=int, default=6, choices=range(1, 11), metavar="1-10")
    encode.add_argument("--zstd-level", type=int, default=10, choices=range(1, 20), metavar="1-19")
    encode.add_argument("--force", action="store_true", help="overwrite output archive")
    encode.set_defaults(func=cmd_encode)

    verify = sub.add_parser("verify", help="verify that the archive restores the target RAW pixels")
    verify.add_argument("keyframe")
    verify.add_argument("target")
    verify.add_argument("archive")
    verify.set_defaults(func=cmd_verify)

    restore = sub.add_parser("restore", help="restore a NEF-like file from keyframe and archive")
    restore.add_argument("keyframe")
    restore.add_argument("archive")
    restore.add_argument("-o", "--output", required=True)
    restore.add_argument("--raw-output", choices=("auto", "nikon", "uncompressed"), default="auto")
    restore.add_argument("--force", action="store_true", help="overwrite output file")
    restore.set_defaults(func=cmd_restore)

    info = sub.add_parser("info", help="print archive metadata")
    info.add_argument("archive")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SpcError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
