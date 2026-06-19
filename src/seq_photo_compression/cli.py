from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from seq_photo_compression.archive import ARCHIVE_EXT, default_archive_path, read_archive, write_archive
from seq_photo_compression.codecs import (
    build_diff,
    encode_jxl_residual,
    restore_raw_from_archive_payload,
    zstd_compress,
    zstd_decompress,
)
from seq_photo_compression.errors import SpcError
from seq_photo_compression.nikon_lossless import derive_nikon_lossless_14_restore_info, encode_nikon_lossless_14
from seq_photo_compression.raw_io import extract_raw_array, raw_to_little_endian_bytes, sha256_file
from seq_photo_compression.tiff import make_zeroed_shell, patch_shell_for_nikon_compressed_raw
from seq_photo_compression.tiff import patch_shell_for_uncompressed_raw


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
        print("diff codec: zstd")
        print(f"diff range: {int(diff_header['min'])}..{int(diff_header['max'])}")
    else:
        print("diff codec: jxl_modular")
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
