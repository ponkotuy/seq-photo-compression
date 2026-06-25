from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from seq_photo_compression.errors import MotionResidualError, SpcError
from seq_photo_compression.external import require_command, run_checked
from seq_photo_compression.motion import (
    encode_residual_pam,
    merge_rggb,
    read_pam,
    restore_from_residual_pam,
    split_rggb,
)


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


def encode_jxl_raw_rggb4(raw: np.ndarray, *, effort: int) -> tuple[bytes, dict]:
    require_command("cjxl")
    planes = split_rggb(raw)
    stacked = np.stack(planes, axis=2)

    with tempfile.TemporaryDirectory(prefix="spc-raw-jxl-") as tmp:
        tmp_dir = Path(tmp)
        raw_pam = tmp_dir / "raw-rggb4.pam"
        raw_jxl = tmp_dir / "raw-rggb4.jxl"
        header = (
            f"P7\nWIDTH {stacked.shape[1]}\nHEIGHT {stacked.shape[0]}\nDEPTH 4\n"
            "MAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        ).encode("ascii")
        raw_pam.write_bytes(header + stacked.astype(">u2", copy=False).tobytes(order="C"))
        run_checked(
            [
                "cjxl",
                str(raw_pam),
                str(raw_jxl),
                "--distance=0",
                "--modular=1",
                f"--effort={effort}",
                "--quiet",
            ]
        )
        payload = raw_jxl.read_bytes()

    metadata = {
        "compression": "jxl_modular",
        "source": "rggb4_raw_u16_pam",
        "jxl_effort": effort,
    }
    return payload, metadata


def decode_jxl_raw_rggb4(header: dict, raw_jxl: bytes) -> np.ndarray:
    require_command("djxl")
    height = int(header["raw"]["height"])
    width = int(header["raw"]["width"])

    with tempfile.TemporaryDirectory(prefix="spc-raw-jxl-") as tmp:
        tmp_dir = Path(tmp)
        raw_jxl_path = tmp_dir / "raw-rggb4.jxl"
        raw_pam_path = tmp_dir / "raw-rggb4.pam"
        raw_jxl_path.write_bytes(raw_jxl)
        run_checked(["djxl", str(raw_jxl_path), str(raw_pam_path), "--quiet"])
        decoded = read_pam(raw_pam_path)

    if decoded.width != width // 2 or decoded.height != height // 2:
        raise SpcError(
            f"decoded RAW geometry mismatch: expected {(height // 2, width // 2)}, "
            f"got {(decoded.height, decoded.width)}"
        )
    planes = [np.ascontiguousarray(decoded.data[:, :, channel]) for channel in range(4)]
    restored = merge_rggb(planes)
    if restored.shape != (height, width):
        raise SpcError(f"restored RAW shape mismatch: expected {(height, width)}, got {restored.shape}")
    return restored


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
