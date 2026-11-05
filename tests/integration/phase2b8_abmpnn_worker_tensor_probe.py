#!/usr/bin/env python3
"""Capture AbMPNN inputs and raw tensors through the production spawn boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pyrosetta as pr  # noqa: E402

from af2_mps_compat import configure_jax  # noqa: E402

if os.environ.get("JAX_PLATFORMS") == "mps":
    JAX = configure_jax("mps")
else:
    JAX = None


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_summary(value) -> dict:
    array = np.asarray(value)
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha_bytes(np.ascontiguousarray(array).tobytes()),
    }
    try:
        summary.update(
            {
                "finite": bool(np.isfinite(array).all()),
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
                "mean": float(np.mean(array)),
                "nan_count": int(np.isnan(array).sum()),
            }
        )
    except (TypeError, ValueError):
        summary.update(
            {
                "finite": None,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "nan_count": None,
            }
        )
    return summary


def child(input_path: str, config_path: str, fixed_positions: str, seed: int):
    jax = JAX or configure_jax("mps")
    import jax.numpy as jnp
    from scipy.special import softmax

    from colabdesign.mpnn.model import mk_mpnn_model

    run_settings = yaml.safe_load(Path(config_path).read_text())["run_settings"]
    model = mk_mpnn_model(
        backbone_noise=run_settings["backbone_noise"],
        model_name=run_settings["model_path"],
        weights=run_settings["mpnn_weights"],
        seed=seed,
    )
    model.prep_inputs(
        pdb_filename=input_path,
        chain="A,B",
        fix_pos=f"{run_settings.get('target_chain', 'A')},{fixed_positions}".rstrip(","),
        rm_aa=run_settings["omit_AAs"],
    )
    inputs = {
        key: array_summary(value)
        for key, value in model._inputs.items()
    }
    output = model.sample(
        temperature=run_settings["sampling_temp"],
        num=1,
        batch=run_settings["num_seqs"],
    )
    logits = np.asarray(output["logits"])
    probability = softmax(logits, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_probability = np.log(probability)
    result = {
        "child_pid": os.getpid(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_checks": bool(jax.config.jax_enable_checks),
        "inputs": inputs,
        "model_params": {
            key: array_summary(value)
            for key, value in model._model.params.items()
        },
        "output": {
            "sequence_count": int(len(output["seq"])),
            "sequences": [str(value) for value in output["seq"]],
            "scores": np.asarray(output["score"]).tolist(),
            "seqid": np.asarray(output["seqid"]).tolist(),
            "logits": array_summary(logits),
            "probabilities": array_summary(probability),
            "log_probabilities": array_summary(log_probability),
            "selected_tokens": array_summary(
                np.asarray(output["S"]).argmax(-1)
            ),
            "decoding_order": array_summary(output["decoding_order"]),
        },
    }
    Path(os.environ["PHASE2B8_OUTPUT"]).write_text(
        json.dumps(result, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dalphaball", required=True)
    parser.add_argument(
        "--fixed-order",
        choices=("hash", "sorted", "reverse-sorted"),
        default="hash",
    )
    args = parser.parse_args()
    jax = JAX or configure_jax("mps")
    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        f"-holes:dalphaball {args.dalphaball} "
        f"-corrections::beta_nov16 true -relax:default_repeats 1"
    )
    import torch

    from germinal.filters.redesign import get_sequence_from_pdb
    from germinal.utils.utils import hotspot_residues

    config = yaml.safe_load(args.config.read_text())
    settings = config["run_settings"]
    chains = get_sequence_from_pdb(str(args.pdb.resolve()))
    length = len(chains["B"])
    framework = [
        f"B{position + 1}"
        for position in range(length)
        if position not in settings["cdr_positions"]
    ]
    interface = hotspot_residues(
        str(args.pdb.resolve()),
        "B",
        target_chain="A",
        atom_distance_cutoff=settings["atom_distance_cutoff"],
    )
    interface_positions = [f"B{position}" for position in interface]
    unique_positions = set(framework + interface_positions)
    if args.fixed_order == "sorted":
        ordered = sorted(
            unique_positions,
            key=lambda value: int(value[1:]),
        )
    elif args.fixed_order == "reverse-sorted":
        ordered = sorted(
            unique_positions,
            key=lambda value: int(value[1:]),
            reverse=True,
        )
    else:
        ordered = list(unique_positions)
    fixed_positions = ",".join(ordered)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PHASE2B8_OUTPUT"] = str(args.output.resolve())
    context = mp.get_context("spawn")
    process = context.Process(
        target=child,
        args=(
            str(args.pdb.resolve()),
            str(args.config.resolve()),
            fixed_positions,
            args.seed,
        ),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        return process.exitcode or 1

    report = json.loads(args.output.read_text())
    report.update(
        {
            "parent_pid": os.getpid(),
            "seed": args.seed,
            "input_pdb": str(args.pdb.resolve()),
            "input_pdb_sha256": sha_bytes(args.pdb.resolve().read_bytes()),
            "fixed_order": args.fixed_order,
            "fixed_positions": ordered,
            "fixed_positions_count": len(ordered),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "torch_mps_available": bool(
                hasattr(torch, "mps") and torch.backends.mps.is_available()
            ),
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": (
            "PASS"
            if len(report["output"]["sequences"]) == settings["num_seqs"]
            and all(np.isfinite(report["output"]["scores"]))
            else "FAIL"
        ),
        "fixed_order": args.fixed_order,
        "fixed_positions_count": len(ordered),
        "logits_finite": report["output"]["logits"]["finite"],
        "probabilities_finite": report["output"]["probabilities"]["finite"],
        "scores": report["output"]["scores"][:5],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
