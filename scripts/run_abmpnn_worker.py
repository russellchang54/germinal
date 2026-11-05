#!/usr/bin/env python3
"""Run one AbMPNN sampling job in a dedicated Python process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax  # noqa: E402


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(
    expected_version: str | None,
    expected_prefix: str | None,
) -> dict:
    import importlib.metadata

    import jax
    import jax_plugins.mps

    distribution = importlib.metadata.distribution("jax-mps")
    plugin_root = Path(jax_plugins.mps.__file__).resolve().parent
    plugin_library = plugin_root / "lib/libpjrt_plugin_mps.dylib"
    identity = {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "jax_mps_version": distribution.version,
        "jax_mps_distribution_path": str(Path(distribution.locate_file("")).resolve()),
        "jax_plugins_mps_path": str(plugin_root),
        "plugin_library_path": str(plugin_library),
        "plugin_library_sha256": _sha256_file(plugin_library),
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
    if expected_version and identity["jax_mps_version"] != expected_version:
        raise RuntimeError(
            "AbMPNN runtime version mismatch: "
            f"expected jax-mps {expected_version}, "
            f"imported {identity['jax_mps_version']} from "
            f"{identity['jax_plugins_mps_path']}"
        )
    if expected_prefix and identity["sys_prefix"] != str(
        Path(expected_prefix).resolve()
    ):
        raise RuntimeError(
            "AbMPNN runtime prefix mismatch: "
            f"expected {Path(expected_prefix).resolve()}, "
            f"got {identity['sys_prefix']}"
        )
    if jax.default_backend() != "mps":
        raise RuntimeError(
            "AbMPNN runtime did not select MPS: "
            f"{jax.default_backend()}"
        )
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--fixed-positions", required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--binder-chain", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-pickle", required=True)
    parser.add_argument("--runtime-identity")
    parser.add_argument("--expected-jax-mps-version")
    parser.add_argument("--expected-prefix")
    args = parser.parse_args()

    platform = os.environ.get("JAX_PLATFORMS")
    if platform in {"mps", "cpu"}:
        configure_jax(platform)
    elif sys.platform == "darwin":
        configure_jax("mps")

    identity = _runtime_identity(
        args.expected_jax_mps_version,
        args.expected_prefix,
    )
    if args.runtime_identity:
        identity_path = Path(args.runtime_identity)
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")

    from germinal.filters.redesign import abmpnn_design

    with open(args.settings_json) as handle:
        run_settings = json.load(handle)
    result = abmpnn_design(
        args.pdb,
        args.fixed_positions,
        run_settings,
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        seed=args.seed,
    )
    with open(args.output_pickle, "wb") as handle:
        pickle.dump(result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
