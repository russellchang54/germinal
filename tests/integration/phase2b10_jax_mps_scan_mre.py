#!/usr/bin/env python3
"""Preserve the upstream jax-mps scan stacked-output fence regression."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "mps")
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "2")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax, random  # noqa: E402


def body(carry, _):
    key, state = carry
    key, subkey = random.split(key)
    state = state + random.normal(subkey, state.shape)
    return (key, state), state


def scan_function(key, state, length):
    return lax.scan(body, (key, state), None, length=length)


def reference_function(key, state, length):
    carry, history = (key, state), []
    for _ in range(length):
        carry, value = body(carry, None)
        history.append(value)
    return carry, jnp.stack(history)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    length, width = 128, 4
    device = jax.devices("mps")[0]
    key = random.PRNGKey(0)
    state = jnp.zeros((width,), jnp.float32, device=device)
    (_, reference_carry), reference_history = jax.jit(
        reference_function,
        static_argnums=2,
        backend="mps",
    )(key, state, length)
    reference_carry = np.asarray(reference_carry)
    reference_history = np.asarray(reference_history)

    failures = []
    attempts = []
    for attempt in range(args.repeats):
        (_, carry), history = jax.jit(
            scan_function,
            static_argnums=2,
            backend="mps",
        )(key, state, length)
        carry = np.asarray(carry)
        history = np.asarray(history)
        rows_close = np.isclose(
            reference_history,
            history,
            rtol=args.tolerance,
            atol=args.tolerance,
        ).all(axis=1)
        carry_close = bool(
            np.allclose(
                reference_carry,
                carry,
                rtol=args.tolerance,
                atol=args.tolerance,
            )
        )
        record = {
            "attempt": attempt + 1,
            "all_history_rows_close": bool(rows_close.all()),
            "carry_close": carry_close,
            "first_bad_row": (
                None if rows_close.all() else int(np.argmax(~rows_close))
            ),
        }
        attempts.append(record)
        if not record["all_history_rows_close"] or not carry_close:
            failures.append(record)

    report = {
        "status": "PASS" if not failures else "FAIL",
        "source": (
            "https://github.com/tillahoffmann/jax-mps/"
            "blob/v0.10.11/tests/test_scan_stacked_output_fence.py"
        ),
        "jax_version": jax.__version__,
        "jax_mps_version": importlib.metadata.version("jax-mps"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(item) for item in jax.devices()],
        "mlx_max_ops_per_buffer": os.environ["MLX_MAX_OPS_PER_BUFFER"],
        "repeats": args.repeats,
        "failures": failures,
        "attempts": attempts,
        "tolerance": args.tolerance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(
        {
            "status": report["status"],
            "jax_mps_version": report["jax_mps_version"],
            "repeats": args.repeats,
            "failure_count": len(failures),
        },
        indent=2,
    ))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
