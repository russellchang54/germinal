#!/usr/bin/env python3
"""Report and validate the selected JAX-MPS runtime identity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--expected-backend", default="mps")
    args = parser.parse_args()

    import jax
    import jax_plugins.mps

    distribution = importlib.metadata.distribution("jax-mps")
    plugin_root = Path(jax_plugins.mps.__file__).resolve().parent
    plugin_library = plugin_root / "lib/libpjrt_plugin_mps.dylib"
    metallib = plugin_root / "lib/mlx.metallib"
    identity = {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "executable": str(Path(sys.executable)),
        "executable_resolved": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "jax_mps_version": distribution.version,
        "jax_mps_distribution_path": str(
            Path(distribution.locate_file("")).resolve()
        ),
        "jax_plugins_mps_path": str(plugin_root),
        "jax_plugins_mps_init_sha256": sha256(plugin_root / "__init__.py"),
        "plugin_library_path": str(plugin_library),
        "plugin_library_sha256": sha256(plugin_library),
        "metallib_path": str(metallib),
        "metallib_sha256": sha256(metallib),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "PYTHONPATH",
                "PYTHONHASHSEED",
                "VIRTUAL_ENV",
                "CONDA_PREFIX",
                "PATH",
            )
        },
    }
    checks = {
        "version": identity["jax_mps_version"] == args.expected_version,
        "prefix": identity["sys_prefix"] == str(args.expected_prefix),
        "backend": identity["jax_default_backend"] == args.expected_backend,
        "plugin_inside_prefix": str(plugin_root).startswith(
            identity["sys_prefix"] + os.sep
        ),
    }
    identity["checks"] = checks
    identity["status"] = "PASS" if all(checks.values()) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))
    return 0 if identity["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
