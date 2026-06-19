from __future__ import annotations

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
