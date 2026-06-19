from __future__ import annotations

import shutil
import subprocess

from seq_photo_compression.errors import SpcError


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
