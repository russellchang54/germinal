#!/usr/bin/env python3
"""Reproduce the production AbMPNN boundary without running AF2 or AF3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Match the production entry point's import order.
import pyrosetta as pr  # noqa: E402

from af2_mps_compat import configure_jax  # noqa: E402

# Spawned workers re-import this module. Match run_germinal.py's module-level
# compatibility setup so the diagnostic exercises the production child path.
if os.environ.get("JAX_PLATFORMS") == "mps":
    JAX = configure_jax("mps")
else:
    JAX = None

# run_germinal.py imports these modules before a spawned AbMPNN worker is
# created. Python's spawn implementation re-imports the entry-point module, so
# these imports are part of the production child's initial state.
from germinal.design.design import germinal_design  # noqa: E402,F401
from germinal.filters import filter_utils, redesign  # noqa: E402,F401
from germinal.utils import config, utils  # noqa: E402,F401
from germinal.utils.io import Trajectory  # noqa: E402,F401


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dalphaball", required=True)
    parser.add_argument(
        "--initialize-pyrosetta",
        action="store_true",
    )
    parser.add_argument(
        "--warm-mps",
        choices=("none", "small", "medium"),
        default="none",
    )
    args = parser.parse_args()

    if not args.initialize_pyrosetta:
        raise SystemExit(
            "--initialize-pyrosetta is required to reproduce the production "
            "module initialization path"
        )

    jax = JAX or configure_jax("mps")
    if args.warm_mps != "none":
        import jax.numpy as jnp

        size = 256 if args.warm_mps == "small" else 2048
        x = jnp.arange(size * size, dtype=jnp.float32).reshape(size, size)
        value = (x @ x).sum()
        value.block_until_ready()
        print(f"MPS warmup result finite: {bool(np.isfinite(float(value)))}")

    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        f"-holes:dalphaball {args.dalphaball} "
        f"-corrections::beta_nov16 true -relax:default_repeats 1"
    )

    from germinal.filters.redesign import run_abmpnn_redesign_pipeline
    from germinal.utils import utils

    config = yaml.safe_load(args.config.read_text())
    run_settings = config["run_settings"]
    utils.clear_memory(clear_jax=False)
    started = time.perf_counter()
    sequences, success = run_abmpnn_redesign_pipeline(
        trajectory_pdb_af=str(args.pdb.resolve()),
        run_settings=run_settings,
        atom_distance_cutoff=run_settings["atom_distance_cutoff"],
        seed=args.seed,
    )
    report = {
        "status": (
            "PASS"
            if success
            and len(sequences) == run_settings["max_mpnn_sequences"]
            and all(
                np.isfinite(float(item["score"])) for item in sequences
            )
            else "FAIL"
        ),
        "seed": args.seed,
        "input_pdb": str(args.pdb.resolve()),
        "input_pdb_sha256": sha256(args.pdb.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "start_method": __import__("multiprocessing").get_start_method(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch_mps_available": bool(
            hasattr(torch, "mps") and torch.backends.mps.is_available()
        ),
        "pyrosetta_initialized": True,
        "mps_warmup": args.warm_mps,
        "success": success,
        "sequence_count": len(sequences),
        "sequences": sequences,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
