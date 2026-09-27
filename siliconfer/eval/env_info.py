"""Run-environment metadata for results/ JSON files.

Every script that writes to results/ should embed this so a number can be
traced back to the exact code, hardware, and library versions that produced
it (git SHA, hardware and OS, library versions, timestamp).
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return bool(out.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_env_info() -> dict:
    """Return a JSON-serializable dict describing the current run environment."""
    from siliconfer.kernels import neon

    mem_bytes = _sysctl("hw.memsize")
    return {
        "kernel_backend": neon._BACKEND,
        "kernel_threads": neon.get_num_threads(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(mem_bytes) / (1024 ** 3), 1) if mem_bytes else None,
        "versions": {
            pkg: _pkg_version(pkg)
            for pkg in ("mlx", "numpy", "torch", "transformers", "datasets", "pybind11")
        },
    }
