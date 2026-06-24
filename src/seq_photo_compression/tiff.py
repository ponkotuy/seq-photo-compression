from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from seq_photo_compression.errors import SpcError


TAG_MAKE = 271
TAG_MODEL = 272
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

    def entry_text(self, entry: TiffEntry) -> str | None:
        if entry.type_id != 2:
            return None
        if entry.data_offset + entry.count > len(self.data):
            raise SpcError("TIFF tag points outside file")
        raw = self.data[entry.data_offset : entry.data_offset + entry.count]
        return raw.rstrip(b"\0").decode("ascii", errors="replace").strip()

    def camera_make_model(self) -> tuple[str | None, str | None]:
        ifd0 = self.parse_ifd(self.first_ifd)
        make = ifd0.entries.get(TAG_MAKE)
        model = ifd0.entries.get(TAG_MODEL)
        return (
            self.entry_text(make) if make is not None else None,
            self.entry_text(model) if model is not None else None,
        )

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


def read_raw_strip_info(nef_path: Path) -> RawStripInfo:
    if not nef_path.is_file():
        raise SpcError(f"NEF not found: {nef_path}")
    data = nef_path.read_bytes()
    parser = TiffParser(data)
    return parser.find_raw_strip()


def read_camera_make_model(nef_path: Path) -> tuple[str | None, str | None]:
    if not nef_path.is_file():
        raise SpcError(f"NEF not found: {nef_path}")
    data = nef_path.read_bytes()
    parser = TiffParser(data)
    return parser.camera_make_model()


def make_zeroed_shell(target_nef: Path) -> tuple[bytes, RawStripInfo]:
    if not target_nef.is_file():
        raise SpcError(f"target NEF not found: {target_nef}")
    data = bytearray(target_nef.read_bytes())
    parser = TiffParser(data)
    raw = parser.find_raw_strip()
    end = raw.strip_offset + raw.strip_byte_count
    if raw.strip_offset < 0 or end > len(data):
        raise SpcError("raw strip points outside target NEF")
    data[raw.strip_offset : end] = b"\0" * raw.strip_byte_count
    return bytes(data), raw


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


def patch_shell_for_nikon_compressed_raw(
    shell: bytes, target_shell_info: dict, raw_info: dict, compressed_raw: bytes
) -> bytes:
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
