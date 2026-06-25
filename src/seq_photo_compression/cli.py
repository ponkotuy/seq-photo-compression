from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seq_photo_compression.archive import (
    ARCHIVE_EXT,
    default_archive_path,
    read_archive,
    read_archive_chunks,
    write_archive,
    write_archive_chunks,
)
from seq_photo_compression.codecs import (
    build_diff,
    decode_jxl_raw_rggb4,
    encode_jxl_raw_rggb4,
    encode_jxl_residual,
    restore_raw_from_archive_payload,
    zstd_compress,
    zstd_decompress,
)
from seq_photo_compression.errors import SpcError
from seq_photo_compression.nikon_lossless import (
    derive_nikon_lossless_14_restore_info,
    encode_nikon_lossless_14,
    get_raw_strip_bytes,
    read_nikon_lossless_14_restore_info_from_makernote,
)
from seq_photo_compression.raw_io import extract_raw_array, raw_to_little_endian_bytes, sha256_file
from seq_photo_compression.tiff import (
    RawStripInfo,
    make_zeroed_shell,
    patch_shell_for_nikon_compressed_raw,
    patch_shell_for_uncompressed_raw,
    read_camera_make_model,
    read_raw_strip_info,
    read_raw_strip_info_from_bytes,
)


@dataclass(frozen=True)
class EncodeOptions:
    diff_codec: str
    motion_mode: str
    jxl_effort: int
    zstd_level: int
    force: bool


@dataclass(frozen=True)
class EncodeResult:
    keyframe: Path
    target: Path
    archive: Path
    target_size: int
    archive_size: int
    raw_width: int
    raw_height: int
    diff_header: dict
    restore_codec: str

    @property
    def ratio(self) -> float:
        return self.archive_size / self.target_size


@dataclass(frozen=True)
class StandaloneEncodeResult:
    target: Path
    archive: Path
    target_size: int
    archive_size: int
    raw_width: int
    raw_height: int
    raw_payload_size: int
    shell_payload_size: int
    raw_codec: str

    @property
    def ratio(self) -> float:
        return self.archive_size / self.target_size


@dataclass(frozen=True)
class StandaloneVerifyResult:
    raw_pixels_equal: bool
    differing_pixels: int
    raw_strip_equal: bool


@dataclass(frozen=True)
class CompatibilitySignature:
    make: str | None
    model: str | None
    raw_width: int
    raw_height: int
    raw_compression: int
    bits_per_sample: int | None


@dataclass(frozen=True)
class NefInspection:
    path: Path
    raw: np.ndarray
    raw_info: RawStripInfo
    signature: CompatibilitySignature


def options_from_args(args: argparse.Namespace) -> EncodeOptions:
    return EncodeOptions(
        diff_codec=args.diff_codec,
        motion_mode=args.motion_mode,
        jxl_effort=args.jxl_effort,
        zstd_level=args.zstd_level,
        force=args.force,
    )


def inspect_nef(path: Path) -> NefInspection:
    raw = extract_raw_array(path)
    raw_info = read_raw_strip_info(path)
    make, model = read_camera_make_model(path)
    signature = CompatibilitySignature(
        make=make,
        model=model,
        raw_width=int(raw.shape[1]),
        raw_height=int(raw.shape[0]),
        raw_compression=raw_info.compression,
        bits_per_sample=raw_info.bits_per_sample,
    )
    return NefInspection(path=path, raw=raw, raw_info=raw_info, signature=signature)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_strip_from_restore_info(raw: np.ndarray, raw_restore_info: dict) -> bytes:
    if raw_restore_info.get("codec") != "nikon_lossless_14":
        raise SpcError(f"unsupported restore RAW codec: {raw_restore_info.get('codec')}")
    return encode_nikon_lossless_14(
        raw,
        raw_restore_info["initial_vpred"],
        padding_bits=int(raw_restore_info.get("padding_bits", 0)),
        padding_value=int(raw_restore_info.get("padding_value", 0)),
        trailing_bytes=bytes.fromhex(str(raw_restore_info.get("trailing_bytes", ""))),
    )


def exact_nikon_restore_info(
    target: Path, raw_info: RawStripInfo, target_raw: np.ndarray, target_strip: bytes
) -> dict | None:
    candidates: list[dict] = []
    makernote_info = read_nikon_lossless_14_restore_info_from_makernote(target, raw_info)
    if makernote_info is not None:
        candidates.append(makernote_info)
    try:
        derived_info = derive_nikon_lossless_14_restore_info(target, raw_info, target_raw)
    except SpcError:
        derived_info = None
    if derived_info is not None:
        candidates.append(derived_info)

    for raw_restore_info in candidates:
        if raw_strip_from_restore_info(target_raw, raw_restore_info) == target_strip:
            return raw_restore_info
    return None


def tiff_patch_header(raw_info: RawStripInfo) -> dict:
    return {
        "compression_entry_pos": raw_info.compression_entry_pos,
        "strip_offset_entry_pos": raw_info.strip_offset_entry_pos,
        "strip_byte_count_entry_pos": raw_info.strip_byte_count_entry_pos,
        "bits_per_sample_entry_pos": raw_info.bits_per_sample_entry_pos,
    }


def encode_archive(
    keyframe: Path,
    target: Path,
    output: Path,
    options: EncodeOptions,
    *,
    base_raw: np.ndarray | None = None,
    target_raw: np.ndarray | None = None,
) -> EncodeResult:
    keyframe = Path(keyframe)
    target = Path(target)
    output = Path(output)

    if base_raw is None:
        base_raw = extract_raw_array(keyframe)
    if target_raw is None:
        target_raw = extract_raw_array(target)

    shell, raw_info = make_zeroed_shell(target)
    nikon_restore_info = read_nikon_lossless_14_restore_info_from_makernote(target, raw_info)
    if nikon_restore_info is None:
        try:
            nikon_restore_info = derive_nikon_lossless_14_restore_info(target, raw_info, target_raw)
        except SpcError:
            nikon_restore_info = None
    shell_zstd = zstd_compress(shell, options.zstd_level)
    if options.diff_codec == "zstd":
        diff = build_diff(base_raw, target_raw)
        diff_payload = zstd_compress(diff.tobytes(order="C"), options.zstd_level)
        diff_header = {
            "compression": "zstd",
            "min": int(diff.min()),
            "max": int(diff.max()),
        }
    elif options.diff_codec == "jxl":
        diff_payload, diff_header = encode_jxl_residual(
            base_raw,
            target_raw,
            motion_mode=options.motion_mode,
            effort=options.jxl_effort,
        )
    else:
        raise SpcError(f"unsupported diff codec: {options.diff_codec}")

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
    write_archive(output, header, shell_zstd, diff_payload, force=options.force)

    return EncodeResult(
        keyframe=keyframe,
        target=target,
        archive=output,
        target_size=target.stat().st_size,
        archive_size=output.stat().st_size,
        raw_width=int(target_raw.shape[1]),
        raw_height=int(target_raw.shape[0]),
        diff_header=diff_header,
        restore_codec="nikon_lossless_14" if nikon_restore_info is not None else "uncompressed",
    )


def encode_standalone_archive(
    target: Path,
    output: Path,
    *,
    jxl_effort: int,
    zstd_level: int,
    fallback: str,
    force: bool,
) -> StandaloneEncodeResult:
    target = Path(target)
    output = Path(output)
    target_raw = extract_raw_array(target)
    shell, raw_info = make_zeroed_shell(target)
    target_strip = get_raw_strip_bytes(target, raw_info)
    raw_restore_info = exact_nikon_restore_info(target, raw_info, target_raw, target_strip)

    shell_zstd = zstd_compress(shell, zstd_level)
    target_shell = {
        "compression": "zstd",
        "original_raw_compression_tag": raw_info.compression,
        "original_bits_per_sample": raw_info.bits_per_sample,
        "raw_restore": raw_restore_info,
        "zeroed_raw_strip_offset": raw_info.strip_offset,
        "zeroed_raw_strip_byte_count": raw_info.strip_byte_count,
        "raw_strip_sha256": sha256_bytes(target_strip),
        "tiff_patch": tiff_patch_header(raw_info),
    }
    header = {
        "version": 2,
        "mode": "standalone_jxl",
        "target": {
            "path": str(target),
            "sha256": sha256_file(target),
            "size": target.stat().st_size,
        },
        "raw": {
            "width": int(target_raw.shape[1]),
            "height": int(target_raw.shape[0]),
            "dtype": "uint16",
        },
        "target_shell": target_shell,
    }

    if raw_restore_info is None:
        if fallback != "raw-strip":
            raise SpcError(
                "target RAW strip cannot be regenerated exactly; pass --fallback raw-strip to store it directly"
            )
        raw_payload = zstd_compress(target_strip, zstd_level)
        header["mode"] = "standalone_raw_strip"
        header["raw_payload"] = {
            "compression": "zstd",
            "source": "original_raw_strip_bytes",
            "zstd_level": zstd_level,
        }
        chunks = {
            "shell_zstd": shell_zstd,
            "raw_strip_zstd": raw_payload,
        }
        raw_codec = "raw_strip_zstd"
    else:
        raw_payload, raw_payload_header = encode_jxl_raw_rggb4(target_raw, effort=jxl_effort)
        header["raw_payload"] = raw_payload_header

        restored_raw = decode_jxl_raw_rggb4(header, raw_payload)
        restored_strip = raw_strip_from_restore_info(restored_raw, raw_restore_info)
        if restored_strip != target_strip:
            raise SpcError("JXL roundtrip did not regenerate the original RAW strip")
        chunks = {
            "shell_zstd": shell_zstd,
            "raw_jxl": raw_payload,
        }
        raw_codec = "jxl_modular_rggb4"

    write_archive_chunks(output, header, chunks, force=force)
    return StandaloneEncodeResult(
        target=target,
        archive=output,
        target_size=target.stat().st_size,
        archive_size=output.stat().st_size,
        raw_width=int(target_raw.shape[1]),
        raw_height=int(target_raw.shape[0]),
        raw_payload_size=len(raw_payload),
        shell_payload_size=len(shell_zstd),
        raw_codec=raw_codec,
    )


def restore_standalone_nef(archive: Path) -> bytes:
    header, chunks = read_archive_chunks(archive)
    mode = header.get("mode")
    if mode not in {"standalone_jxl", "standalone_raw_strip"}:
        raise SpcError(f"archive is not a standalone archive: {mode}")

    try:
        shell = zstd_decompress(chunks["shell_zstd"])
    except KeyError as exc:
        raise SpcError("standalone archive is missing shell_zstd chunk") from exc

    if mode == "standalone_raw_strip":
        try:
            raw_strip = zstd_decompress(chunks["raw_strip_zstd"])
        except KeyError as exc:
            raise SpcError("standalone raw-strip archive is missing raw_strip_zstd chunk") from exc
    else:
        try:
            raw_jxl = chunks["raw_jxl"]
        except KeyError as exc:
            raise SpcError("standalone JXL archive is missing raw_jxl chunk") from exc
        restored_raw = decode_jxl_raw_rggb4(header, raw_jxl)
        raw_restore_info = header["target_shell"].get("raw_restore")
        if raw_restore_info is None:
            raise SpcError("standalone JXL archive is missing RAW restore metadata")
        raw_strip = raw_strip_from_restore_info(restored_raw, raw_restore_info)

    expected_sha256 = header["target_shell"].get("raw_strip_sha256")
    if expected_sha256 is not None and sha256_bytes(raw_strip) != expected_sha256:
        raise SpcError("restored RAW strip SHA-256 does not match archive metadata")
    return patch_shell_for_nikon_compressed_raw(
        shell,
        header["target_shell"],
        header["target_shell"]["tiff_patch"],
        raw_strip,
    )


def verify_standalone_archive(target: Path, archive: Path) -> StandaloneVerifyResult:
    target_raw = extract_raw_array(target)
    restored_nef = restore_standalone_nef(archive)
    target_info = read_raw_strip_info(target)
    restored_info = read_raw_strip_info_from_bytes(restored_nef)
    target_strip = get_raw_strip_bytes(target, target_info)
    restored_strip = restored_nef[
        restored_info.strip_offset : restored_info.strip_offset + restored_info.strip_byte_count
    ]

    header, chunks = read_archive_chunks(archive)
    if header.get("mode") == "standalone_jxl":
        restored_raw = decode_jxl_raw_rggb4(header, chunks["raw_jxl"])
    else:
        with tempfile.TemporaryDirectory(prefix="spc-verify-single-") as tmp:
            restored_path = Path(tmp) / "restored.NEF"
            restored_path.write_bytes(restored_nef)
            restored_raw = extract_raw_array(restored_path)
    raw_pixels_equal = np.array_equal(restored_raw, target_raw)
    differing_pixels = 0 if raw_pixels_equal else int(np.count_nonzero(restored_raw != target_raw))

    return StandaloneVerifyResult(
        raw_pixels_equal=raw_pixels_equal,
        differing_pixels=differing_pixels,
        raw_strip_equal=restored_strip == target_strip,
    )


def print_encode_result(result: EncodeResult) -> None:
    print(f"archive: {result.archive}")
    print(f"target NEF size: {result.target_size:,} bytes")
    print(f"archive size: {result.archive_size:,} bytes ({result.ratio:.2%} of target)")
    print(f"raw shape: {result.raw_width}x{result.raw_height}")
    if result.diff_header["compression"] == "zstd":
        print("diff codec: zstd")
        print(f"diff range: {int(result.diff_header['min'])}..{int(result.diff_header['max'])}")
    else:
        print("diff codec: jxl_modular")
        print(f"motion mode: {result.diff_header['motion_mode']}")
        print(f"motion status: {result.diff_header['motion_status']}")
        print(f"motion score: {result.diff_header['motion_score']:.6f}")
        print(f"residual range: {result.diff_header['residual_min']}..{result.diff_header['residual_max']}")
        print(f"jxl effort: {result.diff_header['jxl_effort']}")
    print(f"restore RAW codec: {result.restore_codec}")


def print_standalone_encode_result(result: StandaloneEncodeResult) -> None:
    print(f"archive: {result.archive}")
    print(f"target NEF size: {result.target_size:,} bytes")
    print(f"archive size: {result.archive_size:,} bytes ({result.ratio:.2%} of target)")
    print(f"raw shape: {result.raw_width}x{result.raw_height}")
    print(f"raw codec: {result.raw_codec}")
    print(f"raw payload size: {result.raw_payload_size:,} bytes")
    print(f"shell payload size: {result.shell_payload_size:,} bytes")


def cmd_encode(args: argparse.Namespace) -> None:
    target = Path(args.target)
    output = Path(args.output) if args.output else default_archive_path(target)
    result = encode_archive(Path(args.keyframe), target, output, options_from_args(args))
    print_encode_result(result)


def cmd_encode_single(args: argparse.Namespace) -> None:
    target = Path(args.target)
    output = Path(args.output) if args.output else default_archive_path(target)
    result = encode_standalone_archive(
        target,
        output,
        jxl_effort=args.jxl_effort,
        zstd_level=args.zstd_level,
        fallback=args.fallback,
        force=args.force,
    )
    print_standalone_encode_result(result)


def verify_raw_pixels(
    keyframe: Path,
    target: Path,
    archive: Path,
    *,
    base_raw: np.ndarray | None = None,
    target_raw: np.ndarray | None = None,
) -> tuple[bool, int]:
    keyframe = Path(keyframe)
    target = Path(target)
    archive = Path(archive)

    header, _shell_zstd, diff_payload = read_archive(archive)
    if sha256_file(keyframe) != header["keyframe"]["sha256"]:
        raise SpcError("keyframe SHA-256 does not match archive metadata")

    if base_raw is None:
        base_raw = extract_raw_array(keyframe)
    if target_raw is None:
        target_raw = extract_raw_array(target)
    restored = restore_raw_from_archive_payload(header, base_raw, diff_payload)
    equal = np.array_equal(restored, target_raw)
    differing = 0 if equal else int(np.count_nonzero(restored != target_raw))
    return equal, differing


def cmd_verify(args: argparse.Namespace) -> None:
    equal, differing = verify_raw_pixels(Path(args.keyframe), Path(args.target), Path(args.archive))
    print(f"raw match: {equal}")
    print(f"differing pixels: {differing}")
    if not equal:
        raise SpcError("restored RAW does not match target RAW")


def cmd_verify_single(args: argparse.Namespace) -> None:
    result = verify_standalone_archive(Path(args.target), Path(args.archive))
    print(f"raw pixels match: {result.raw_pixels_equal}")
    print(f"differing pixels: {result.differing_pixels}")
    print(f"raw strip match: {result.raw_strip_equal}")
    if not result.raw_pixels_equal:
        raise SpcError("restored RAW pixels do not match target RAW")
    if not result.raw_strip_equal:
        raise SpcError("restored RAW strip does not match target NEF")


def iter_nef_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SpcError(f"directory not found: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".nef")
    if not files:
        raise SpcError(f"no NEF files found in directory: {directory}")
    return files


def print_encode_dir_summary(
    *,
    original_size: int,
    stored_size: int,
    keyframe_count: int,
    archive_count: int,
    failed_count: int,
    verified_count: int,
    verify_failed_count: int,
) -> None:
    ratio = stored_size / original_size if original_size else 0.0
    saved = original_size - stored_size
    print("summary:")
    print(f"original NEF total: {original_size:,} bytes")
    print(f"storage set total: {stored_size:,} bytes ({ratio:.2%} of original)")
    print(f"saved: {saved:,} bytes")
    print(f"keyframes: {keyframe_count}")
    print(f"archives: {archive_count}")
    print(f"failed: {failed_count}")
    if verified_count or verify_failed_count:
        print(f"verified: {verified_count}")
        print(f"verify failed: {verify_failed_count}")


def cmd_encode_dir(args: argparse.Namespace) -> None:
    directory = Path(args.directory)
    options = options_from_args(args)
    max_archive_ratio = args.max_archive_ratio
    if max_archive_ratio <= 0 or max_archive_ratio > 1:
        raise SpcError("--max-archive-ratio must be greater than 0 and less than or equal to 1")
    files = iter_nef_files(directory)
    keyframes: dict[CompatibilitySignature, NefInspection] = {}

    original_size = 0
    stored_size = 0
    keyframe_count = 0
    archive_count = 0
    failed_count = 0
    verified_count = 0
    verify_failed_count = 0

    for nef in files:
        target_size = nef.stat().st_size
        original_size += target_size
        inspection: NefInspection | None = None
        try:
            inspection = inspect_nef(nef)
            keyframe = keyframes.get(inspection.signature)
            if keyframe is None:
                keyframes[inspection.signature] = inspection
                keyframe_count += 1
                stored_size += target_size
                print(f"keyframe: {nef.name} ({target_size:,} bytes)")
                continue

            output = default_archive_path(nef)
            result = encode_archive(
                keyframe.path,
                nef,
                output,
                options,
                base_raw=keyframe.raw,
                target_raw=inspection.raw,
            )

            if result.ratio >= max_archive_ratio:
                result.archive.unlink(missing_ok=True)
                keyframes[inspection.signature] = inspection
                keyframe_count += 1
                stored_size += target_size
                print(
                    f"ratio-keyframe: {nef.name} "
                    f"archive={result.archive_size:,} original={target_size:,} "
                    f"ratio={result.ratio:.2%} threshold={max_archive_ratio:.2%}"
                )
                continue

            if args.verify:
                equal, differing = verify_raw_pixels(
                    keyframe.path,
                    nef,
                    result.archive,
                    base_raw=keyframe.raw,
                    target_raw=inspection.raw,
                )
                if not equal:
                    result.archive.unlink(missing_ok=True)
                    keyframes[inspection.signature] = inspection
                    keyframe_count += 1
                    failed_count += 1
                    verify_failed_count += 1
                    stored_size += target_size
                    print(f"failed: {nef.name} verify mismatch differing_pixels={differing:,}")
                    continue
                verified_count += 1

            archive_count += 1
            stored_size += result.archive_size
            verify_suffix = " verified=True" if args.verify else ""
            print(
                f"encoded: {nef.name} -> {result.archive.name} "
                f"{result.archive_size:,}/{target_size:,} ({result.ratio:.2%}){verify_suffix}"
            )
        except SpcError as exc:
            failed_count += 1
            stored_size += target_size
            if inspection is not None:
                keyframes[inspection.signature] = inspection
                keyframe_count += 1
            print(f"failed: {nef.name}: {exc}")

    print_encode_dir_summary(
        original_size=original_size,
        stored_size=stored_size,
        keyframe_count=keyframe_count,
        archive_count=archive_count,
        failed_count=failed_count,
        verified_count=verified_count,
        verify_failed_count=verify_failed_count,
    )


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


def cmd_restore_single(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SpcError(f"output exists, pass --force to overwrite: {output}")

    restored_nef = restore_standalone_nef(archive)
    output.write_bytes(restored_nef)
    print(f"restored NEF file: {output}")
    print(f"output size: {output.stat().st_size:,} bytes")
    print("restore RAW codec: original_nikon_strip")


def cmd_info(args: argparse.Namespace) -> None:
    header, _chunks = read_archive_chunks(Path(args.archive))
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

    encode_single = sub.add_parser("encode-single", help="create a standalone lossless archive for one NEF")
    encode_single.add_argument("target", help="NEF to encode as a standalone custom format")
    encode_single.add_argument("-o", "--output", help=f"output archive path, default: TARGET.NEF{ARCHIVE_EXT}")
    encode_single.add_argument("--jxl-effort", type=int, default=6, choices=range(1, 11), metavar="1-10")
    encode_single.add_argument("--zstd-level", type=int, default=10, choices=range(1, 20), metavar="1-19")
    encode_single.add_argument(
        "--fallback",
        choices=("fail", "raw-strip"),
        default="fail",
        help="what to do when the RAW strip cannot be regenerated exactly",
    )
    encode_single.add_argument("--force", action="store_true", help="overwrite output archive")
    encode_single.set_defaults(func=cmd_encode_single)

    encode_dir = sub.add_parser("encode-dir", help="encode NEF files in a directory as a storage set")
    encode_dir.add_argument("directory", help="directory containing NEF files")
    encode_dir.add_argument("--diff-codec", choices=("zstd", "jxl"), default="jxl")
    encode_dir.add_argument("--motion-mode", choices=("none", "translation", "ecc_affine"), default="ecc_affine")
    encode_dir.add_argument("--jxl-effort", type=int, default=6, choices=range(1, 11), metavar="1-10")
    encode_dir.add_argument("--zstd-level", type=int, default=10, choices=range(1, 20), metavar="1-19")
    encode_dir.add_argument(
        "--max-archive-ratio",
        type=float,
        default=0.9,
        help="make the target a new keyframe when archive/original is at or above this ratio",
    )
    encode_dir.add_argument("--force", action="store_true", help="overwrite output archives")
    encode_dir.add_argument("--verify", action="store_true", help="verify RAW pixels for each created archive")
    encode_dir.set_defaults(func=cmd_encode_dir)

    verify = sub.add_parser("verify", help="verify that the archive restores the target RAW pixels")
    verify.add_argument("keyframe")
    verify.add_argument("target")
    verify.add_argument("archive")
    verify.set_defaults(func=cmd_verify)

    verify_single = sub.add_parser("verify-single", help="verify a standalone archive against its source NEF")
    verify_single.add_argument("target")
    verify_single.add_argument("archive")
    verify_single.set_defaults(func=cmd_verify_single)

    restore = sub.add_parser("restore", help="restore a NEF-like file from keyframe and archive")
    restore.add_argument("keyframe")
    restore.add_argument("archive")
    restore.add_argument("-o", "--output", required=True)
    restore.add_argument("--raw-output", choices=("auto", "nikon", "uncompressed"), default="auto")
    restore.add_argument("--force", action="store_true", help="overwrite output file")
    restore.set_defaults(func=cmd_restore)

    restore_single = sub.add_parser("restore-single", help="restore a NEF file from a standalone archive")
    restore_single.add_argument("archive")
    restore_single.add_argument("-o", "--output", required=True)
    restore_single.add_argument("--force", action="store_true", help="overwrite output file")
    restore_single.set_defaults(func=cmd_restore_single)

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
