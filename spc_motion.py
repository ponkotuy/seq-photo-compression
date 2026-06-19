from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CHANNELS = 4
RESIDUAL_OFFSET = 32768


class MotionResidualError(RuntimeError):
    pass


@dataclass(frozen=True)
class MotionResidualStats:
    status: str
    score: float
    matrix: str
    residual_min: int
    residual_max: int
    offset: int = RESIDUAL_OFFSET


@dataclass(frozen=True)
class PamImage:
    width: int
    height: int
    depth: int
    maxval: int
    data: np.ndarray


def validate_raw_geometry(raw: np.ndarray) -> tuple[int, int]:
    if raw.ndim != 2:
        raise MotionResidualError(f"RAW array must be 2-dimensional: {raw.shape}")
    height, width = raw.shape
    if width <= 0 or height <= 0 or width % 2 != 0 or height % 2 != 0:
        raise MotionResidualError("RAW dimensions must be positive even numbers")
    return width, height


def split_rggb(raw: np.ndarray) -> list[np.ndarray]:
    validate_raw_geometry(raw)
    return [
        np.ascontiguousarray(raw[0::2, 0::2], dtype=np.uint16),
        np.ascontiguousarray(raw[0::2, 1::2], dtype=np.uint16),
        np.ascontiguousarray(raw[1::2, 0::2], dtype=np.uint16),
        np.ascontiguousarray(raw[1::2, 1::2], dtype=np.uint16),
    ]


def merge_rggb(planes: list[np.ndarray]) -> np.ndarray:
    if len(planes) != CHANNELS:
        raise MotionResidualError(f"expected {CHANNELS} RGGB planes, got {len(planes)}")
    plane_height, plane_width = planes[0].shape
    raw = np.empty((plane_height * 2, plane_width * 2), dtype=np.uint16)
    raw[0::2, 0::2] = planes[0]
    raw[0::2, 1::2] = planes[1]
    raw[1::2, 0::2] = planes[2]
    raw[1::2, 1::2] = planes[3]
    return raw


def make_alignment_image(planes: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(planes[0].shape, dtype=np.float32)
    for plane in planes:
        acc += plane.astype(np.float32)
    acc *= 0.25
    return acc


def identity_matrix() -> np.ndarray:
    return np.eye(2, 3, dtype=np.float32)


def matrix_to_string(matrix: np.ndarray) -> str:
    rows = []
    for row in matrix:
        rows.append(",".join(f"{float(value):.9g}" for value in row))
    return ";".join(rows)


def parse_affine_matrix(text: str) -> np.ndarray:
    rows = text.split(";")
    if len(rows) != 2:
        raise MotionResidualError("invalid affine matrix")
    matrix = np.empty((2, 3), dtype=np.float32)
    for y, row in enumerate(rows):
        cols = row.split(",")
        if len(cols) != 3:
            raise MotionResidualError("invalid affine matrix")
        for x, value in enumerate(cols):
            matrix[y, x] = float(value)
    return matrix


def warp_plane(source: np.ndarray, matrix: np.ndarray, motion_mode: int) -> np.ndarray:
    if motion_mode == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(
            source,
            matrix,
            (source.shape[1], source.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return cv2.warpAffine(
        source,
        matrix,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def build_predictor(
    base_planes: list[np.ndarray],
    matrix: np.ndarray,
    motion_mode: int,
) -> list[np.ndarray]:
    return [warp_plane(plane, matrix, motion_mode) for plane in base_planes]


def estimate_motion(
    base_planes: list[np.ndarray],
    target_planes: list[np.ndarray],
    mode: str,
) -> tuple[np.ndarray, int, float, str]:
    warp_matrix = identity_matrix()
    motion_mode = cv2.MOTION_AFFINE
    score = 0.0
    status = "ok"

    if mode == "none":
        return warp_matrix, cv2.MOTION_TRANSLATION, score, status

    base_align = make_alignment_image(base_planes)
    target_align = make_alignment_image(target_planes)
    if mode == "translation":
        shift, response = cv2.phaseCorrelate(base_align, target_align)
        warp_matrix[0, 2] = np.float32(shift[0])
        warp_matrix[1, 2] = np.float32(shift[1])
        return warp_matrix, cv2.MOTION_TRANSLATION, float(response), status

    if mode == "ecc_affine":
        base_norm = cv2.normalize(base_align, None, 0.0, 1.0, cv2.NORM_MINMAX)
        target_norm = cv2.normalize(target_align, None, 0.0, 1.0, cv2.NORM_MINMAX)
        try:
            score, warp_matrix = cv2.findTransformECC(
                target_norm,
                base_norm,
                warp_matrix,
                cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-7),
                None,
                5,
            )
        except cv2.error as exc:
            status = f"ecc_failed: {exc}"
            warp_matrix = identity_matrix()
            score = 0.0
        return warp_matrix, cv2.MOTION_AFFINE, float(score), status

    raise MotionResidualError(f"unknown mode: {mode}")


def write_pam_residual(
    path: Path,
    target_planes: list[np.ndarray],
    predictor: list[np.ndarray],
) -> tuple[int, int]:
    residual_planes = [
        target.astype(np.int32) - pred.astype(np.int32) for target, pred in zip(target_planes, predictor, strict=True)
    ]
    residual = np.stack(residual_planes, axis=2)
    residual_min = int(residual.min())
    residual_max = int(residual.max())
    coded = residual + RESIDUAL_OFFSET
    if int(coded.min()) < 0 or int(coded.max()) > 65535:
        raise MotionResidualError("residual outside uint16 offset range")

    plane_height, plane_width = target_planes[0].shape
    header = (
        f"P7\nWIDTH {plane_width}\nHEIGHT {plane_height}\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
    ).encode("ascii")
    path.write_bytes(header + coded.astype(">u2").tobytes(order="C"))
    return residual_min, residual_max


def read_pam(path: Path) -> PamImage:
    with path.open("rb") as f:
        first_line = f.readline().rstrip(b"\n")
        if first_line != b"P7":
            raise MotionResidualError("unsupported PAM magic")

        values: dict[str, int] = {}
        while True:
            line = f.readline()
            if not line:
                raise MotionResidualError("unexpected EOF in PAM header")
            line = line.rstrip(b"\n")
            if line == b"ENDHDR":
                break
            parts = line.split()
            if len(parts) >= 2 and parts[0] in {b"WIDTH", b"HEIGHT", b"DEPTH", b"MAXVAL"}:
                values[parts[0].decode("ascii")] = int(parts[1])
        width = values.get("WIDTH", 0)
        height = values.get("HEIGHT", 0)
        depth = values.get("DEPTH", 0)
        maxval = values.get("MAXVAL", 0)
        if width <= 0 or height <= 0 or depth != CHANNELS or maxval != 65535:
            raise MotionResidualError("unsupported PAM geometry")
        data = f.read()

    expected = width * height * depth * np.dtype(">u2").itemsize
    if len(data) != expected:
        raise MotionResidualError(f"PAM data size mismatch: expected {expected}, got {len(data)}")
    arr = np.frombuffer(data, dtype=">u2").astype(np.uint16, copy=True).reshape((height, width, depth))
    return PamImage(width, height, depth, maxval, arr)


def encode_residual_pam(
    base_raw: np.ndarray,
    target_raw: np.ndarray,
    *,
    motion_mode: str,
    output: Path,
) -> MotionResidualStats:
    if base_raw.shape != target_raw.shape:
        raise MotionResidualError(f"RAW shape mismatch: base={base_raw.shape}, target={target_raw.shape}")
    validate_raw_geometry(target_raw)

    base_planes = split_rggb(base_raw)
    target_planes = split_rggb(target_raw)
    warp_matrix, cv_motion_mode, score, status = estimate_motion(base_planes, target_planes, motion_mode)
    predictor = base_planes if motion_mode == "none" else build_predictor(base_planes, warp_matrix, cv_motion_mode)
    residual_min, residual_max = write_pam_residual(output, target_planes, predictor)
    return MotionResidualStats(
        status=status,
        score=score,
        matrix=matrix_to_string(warp_matrix),
        residual_min=residual_min,
        residual_max=residual_max,
    )


def restore_from_residual_pam(
    base_raw: np.ndarray,
    residual_pam: Path,
    *,
    motion_mode: str,
    matrix: str,
) -> np.ndarray:
    validate_raw_geometry(base_raw)
    base_planes = split_rggb(base_raw)
    warp_matrix = parse_affine_matrix(matrix)
    if motion_mode in {"none", "translation"}:
        cv_motion_mode = cv2.MOTION_TRANSLATION
    elif motion_mode == "ecc_affine":
        cv_motion_mode = cv2.MOTION_AFFINE
    else:
        raise MotionResidualError(f"unknown mode: {motion_mode}")

    predictor = base_planes if motion_mode == "none" else build_predictor(base_planes, warp_matrix, cv_motion_mode)
    residual = read_pam(residual_pam)
    if residual.width != base_raw.shape[1] // 2 or residual.height != base_raw.shape[0] // 2:
        raise MotionResidualError("PAM dimensions do not match RAW geometry")

    predictor_stack = np.stack([plane.astype(np.int32) for plane in predictor], axis=2)
    residual_values = residual.data.astype(np.int32) - RESIDUAL_OFFSET
    restored = predictor_stack + residual_values
    if int(restored.min()) < 0 or int(restored.max()) > 65535:
        raise MotionResidualError("restored RAW value outside uint16 range")
    restored_planes = [np.ascontiguousarray(restored[:, :, channel].astype(np.uint16)) for channel in range(CHANNELS)]
    return merge_rggb(restored_planes)
