from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from seq_photo_compression.errors import SpcError
from seq_photo_compression.tiff import RawStripInfo, TIFF_COMPRESSION_NIKON_NEF


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

TAG_EXIF_IFD = 0x8769
TAG_MAKER_NOTE = 0x927C
TAG_NIKON_COMPRESSION_INFO = 0x0096

TYPE_LONG = 4
TYPE_SIZES = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    7: 1,  # UNDEFINED
    8: 2,  # SSHORT
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
    13: 4,  # IFD
}


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
NIKON_LOSSLESS_14_CODE_BITS_BY_LENGTH = np.array(
    [NIKON_LOSSLESS_14_CODES[length][1] for length in range(15)],
    dtype=np.uint8,
)
UINT14_BIT_LENGTHS = np.array([value.bit_length() for value in range(0x4000 + 1)], dtype=np.uint8)


def _tiff_endian(data: bytes, offset: int) -> str:
    if data[offset : offset + 2] == b"II":
        return "<"
    if data[offset : offset + 2] == b"MM":
        return ">"
    raise SpcError("not a TIFF byte order marker")


def _u16(data: bytes, endian: str, offset: int) -> int:
    return struct.unpack_from(endian + "H", data, offset)[0]


def _u32(data: bytes, endian: str, offset: int) -> int:
    return struct.unpack_from(endian + "I", data, offset)[0]


def _read_ifd_entries(data: bytes, base: int, ifd_offset: int, endian: str) -> dict[int, tuple[int, int, int, int]]:
    absolute = base + ifd_offset
    if absolute <= 0 or absolute + 2 > len(data):
        raise SpcError(f"invalid TIFF IFD offset: {ifd_offset}")
    count = _u16(data, endian, absolute)
    entries: dict[int, tuple[int, int, int, int]] = {}
    pos = absolute + 2
    if pos + count * 12 + 4 > len(data):
        raise SpcError("TIFF IFD points outside file")
    for _ in range(count):
        tag = _u16(data, endian, pos)
        type_id = _u16(data, endian, pos + 2)
        value_count = _u32(data, endian, pos + 4)
        value_offset_pos = pos + 8
        size = TYPE_SIZES.get(type_id, 1) * value_count
        data_offset = value_offset_pos if size <= 4 else base + _u32(data, endian, value_offset_pos)
        if data_offset < 0 or data_offset + size > len(data):
            raise SpcError("TIFF tag points outside file")
        entries[tag] = (type_id, value_count, data_offset, value_offset_pos)
        pos += 12
    return entries


def _entry_u32(data: bytes, endian: str, entry: tuple[int, int, int, int]) -> int:
    type_id, value_count, data_offset, _value_offset_pos = entry
    if type_id != TYPE_LONG or value_count < 1:
        raise SpcError("TIFF tag is not a LONG value")
    return _u32(data, endian, data_offset)


def read_nikon_lossless_14_restore_info_from_makernote(
    target_nef: Path,
    raw_info: RawStripInfo,
) -> dict | None:
    if raw_info.compression != TIFF_COMPRESSION_NIKON_NEF or raw_info.bits_per_sample != 14:
        return None

    try:
        data = target_nef.read_bytes()
        endian = _tiff_endian(data, 0)
        if _u16(data, endian, 2) != 42:
            return None
        ifd0 = _read_ifd_entries(data, 0, _u32(data, endian, 4), endian)
        exif_entry = ifd0.get(TAG_EXIF_IFD)
        if exif_entry is None:
            return None
        exif = _read_ifd_entries(data, 0, _entry_u32(data, endian, exif_entry), endian)
        maker_entry = exif.get(TAG_MAKER_NOTE)
        if maker_entry is None:
            return None

        _maker_type, maker_count, maker_offset, _maker_value_pos = maker_entry
        if maker_count < 16 or data[maker_offset : maker_offset + 6] != b"Nikon\0":
            return None
        maker_base = maker_offset + 10
        maker_endian = _tiff_endian(data, maker_base)
        if _u16(data, maker_endian, maker_base + 2) != 42:
            return None
        maker_ifd = _read_ifd_entries(data, maker_base, _u32(data, maker_endian, maker_base + 4), maker_endian)
        compression_entry = maker_ifd.get(TAG_NIKON_COMPRESSION_INFO)
        if compression_entry is None:
            return None

        _type_id, value_count, compression_offset, _value_offset_pos = compression_entry
        if value_count < 10:
            return None
        ver0 = data[compression_offset]
        ver1 = data[compression_offset + 1]
        pos = compression_offset + 2
        if ver0 == 0x49 or ver1 == 0x58:
            pos += 2110
        # D850 lossless 14-bit NEFs use the F0 table. Other Nikon variants
        # may need curves or split tables, so keep them on the old safe path.
        if ver0 != 0x46 or pos + 10 > compression_offset + value_count:
            return None

        flat_vpred = struct.unpack_from(maker_endian + "4H", data, pos)
        if any(value > 0x3FFF for value in flat_vpred):
            return None
        return {
            "codec": "nikon_lossless_14",
            "compression_tag": TIFF_COMPRESSION_NIKON_NEF,
            "bits_per_sample": 14,
            "initial_vpred": [
                [int(flat_vpred[0]), int(flat_vpred[1])],
                [int(flat_vpred[2]), int(flat_vpred[3])],
            ],
            "source": "nikon_makernote_compression_info",
        }
    except (IndexError, OSError, SpcError, struct.error):
        return None


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


def complete_nikon_lossless_14_restore_info(
    raw: np.ndarray,
    restore_info: dict,
    original_strip: bytes,
    *,
    chunk_rows: int = 256,
) -> dict:
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16, copy=False)
    if restore_info.get("codec") != "nikon_lossless_14":
        raise SpcError(f"unsupported restore RAW codec: {restore_info.get('codec')}")
    if int(raw.max()) > 0x3FFF:
        raise SpcError("Nikon lossless 14-bit output requires RAW values in 0..16383")

    height, width = raw.shape
    initial_vpred = restore_info["initial_vpred"]
    total_bits = 0

    for start in range(0, height, chunk_rows):
        end = min(start + chunk_rows, height)
        rows = np.arange(start, end)
        diff = np.empty((end - start, width), dtype=np.int32)

        diff[:, 0] = raw[start:end, 0].astype(np.int32)
        diff[:, 1] = raw[start:end, 1].astype(np.int32)
        first_even = rows % 2 == 0
        has_previous_same_parity = rows >= 2
        diff[first_even & ~has_previous_same_parity, 0] -= int(initial_vpred[0][0])
        diff[~first_even & ~has_previous_same_parity, 0] -= int(initial_vpred[1][0])
        diff[first_even & ~has_previous_same_parity, 1] -= int(initial_vpred[0][1])
        diff[~first_even & ~has_previous_same_parity, 1] -= int(initial_vpred[1][1])
        if np.any(has_previous_same_parity):
            previous_rows = rows[has_previous_same_parity] - 2
            diff[has_previous_same_parity, 0] -= raw[previous_rows, 0].astype(np.int32)
            diff[has_previous_same_parity, 1] -= raw[previous_rows, 1].astype(np.int32)

        diff[:, 2:] = raw[start:end, 2:].astype(np.int32) - raw[start:end, :-2].astype(np.int32)
        abs_diff = np.abs(diff)
        if int(abs_diff.max()) > 0x3FFF:
            raise SpcError("Nikon lossless diff does not fit 14 bits")
        lengths = UINT14_BIT_LENGTHS[abs_diff]
        total_bits += int((NIKON_LOSSLESS_14_CODE_BITS_BY_LENGTH[lengths] + lengths).sum())

    encoded_byte_count = (total_bits + 7) // 8
    if encoded_byte_count > len(original_strip):
        raise SpcError("computed Nikon lossless bitstream is longer than the original RAW strip")
    padding_bits = encoded_byte_count * 8 - total_bits
    padding_value = original_strip[encoded_byte_count - 1] & ((1 << padding_bits) - 1) if padding_bits else 0

    completed = dict(restore_info)
    completed["padding_bits"] = int(padding_bits)
    completed["padding_value"] = int(padding_value)
    completed["trailing_bytes"] = original_strip[encoded_byte_count:].hex()
    completed["padding_source"] = "computed_bit_length"
    return completed


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
