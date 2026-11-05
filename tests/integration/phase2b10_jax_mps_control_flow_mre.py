#!/usr/bin/env python3
"""Preserve the upstream jax-mps #195 nested-control-flow correctness MRE."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "mps")

import jax
import jax.numpy as jnp
from jax import lax


EXPECTED_GRADIENT = 1.001**65


def function(x):
    return lax.fori_loop(
        0,
        130,
        lambda index, value: lax.cond(
            index % 2 == 0,
            lambda inner: inner + 0.01,
            lambda inner: inner * 1.001,
            value,
        ),
        x,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    x = jnp.float32(1.0)
    eager = float(jax.grad(function)(x))
    jitted = float(jax.jit(jax.grad(function))(x))
    eager_error = abs(eager - EXPECTED_GRADIENT)
    jitted_error = abs(jitted - EXPECTED_GRADIENT)
    report = {
        "status": (
            "PASS"
            if eager_error <= args.tolerance
            and jitted_error <= args.tolerance
            else "FAIL"
        ),
        "source": "https://github.com/tillahoffmann/jax-mps/issues/195",
        "jax_version": jax.__version__,
        "jax_mps_version": importlib.metadata.version("jax-mps"),
        "jax_plugins_mps_file": str(
            Path(jax_plugins_file()).resolve()
        ),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "expected_gradient": EXPECTED_GRADIENT,
        "eager_gradient": eager,
        "jitted_gradient": jitted,
        "eager_absolute_error": eager_error,
        "jitted_absolute_error": jitted_error,
        "tolerance": args.tolerance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


def jax_plugins_file() -> str:
    import jax_plugins.mps

    return jax_plugins.mps.__file__


if __name__ == "__main__":
    raise SystemExit(main())
