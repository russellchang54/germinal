#!/usr/bin/env python3
"""Recreate the production AF2 parent state, then exercise actual AbMPNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Match run_germinal.py's production import order.
import pyrosetta as pr  # noqa: E402

from af2_mps_compat import configure_jax  # noqa: E402

JAX = configure_jax("mps") if os.environ.get("JAX_PLATFORMS") == "mps" else None

from germinal.design.design import germinal_design  # noqa: E402
from germinal.filters import redesign  # noqa: E402
from germinal.utils import config, utils  # noqa: E402
from germinal.utils.io import Trajectory  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "docs/porting/af3_mps_probe/results/phase2b8/state_reproduction"
        ),
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = REPO_ROOT / "configs"
    overrides = [
        "run=vhh_il3_macos",
        "target=il3",
        "filter/initial=vhh_il3",
        "filter/final=vhh_il3",
        f"project_dir={output_root}",
        "pdb_dir=docs/porting/af3_mps_probe/results/phase2b4/run1/input",
        (
            "+starting_pdb_complex=docs/porting/af3_mps_probe/results/"
            "phase2b4/run1/input/il3_11_8_18_nb.pdb"
        ),
        (
            "af_params_dir=.envs/germinal-mps-design/share/"
            "af2-params/extracted"
        ),
        "dssp_path=.envs/germinal-macos/bin/mkdssp",
        "dalphaball_path=.envs/germinal-macos/bin/dalphaball",
        "af3_database_manifest=null",
        "af3_db_dir=/Volumes/T9/database",
        "pregenerate_seeds=true",
        "max_trajectories=1",
        "max_hallucinated_trajectories=1",
        "max_passing_designs=1",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
    ):
        cfg = compose(config_name="config", overrides=overrides)
    processed = config.process_config(cfg)
    run_settings, target_settings, _, _ = processed.values()
    io, run_settings = config.initialize_germinal_run(
        run_settings, target_settings
    )
    io.save_run_config(run_settings, target_settings)
    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        f"-holes:dalphaball {run_settings['dalphaball_path']} "
        f"-corrections::beta_nov16 true -relax:default_repeats 1"
    )

    np.random.seed(1)
    seeds = [
        np.random.randint(2**32 - 1)
        for _ in range(run_settings["max_trajectories"])
    ]
    seed = int(seeds[0])
    design_name = f"{target_settings['target_name']}_{run_settings['type']}_s{seed}"
    trajectory = Trajectory(
        design_name,
        run_settings["experiment_name"],
        ",".join(map(str, run_settings["cdr_lengths"])),
        target_settings["target_hotspots"],
    )
    trajectory.set_save_location("trajectories")

    started = time.perf_counter()
    design_output = germinal_design(
        design_name, run_settings, target_settings, io, seed
    )
    af2_seconds = time.perf_counter() - started
    trajectory_sequence = design_output._tmp["best"]["seq"]
    trajectory_pdb_af = (
        io.layout.trajectories / f"structures/{design_name}.pdb"
    )

    boundary_source = (
        REPO_ROOT
        / "docs/porting/af3_mps_probe/results/phase2b7/"
        "abmpnn_diagnosis/run2_in_pipeline_abmpnn_input.pdb"
    )
    shutil.copyfile(boundary_source, trajectory_pdb_af)
    if hasattr(design_output, "ablm_model") and design_output.ablm_model is not None:
        design_output.ablm_model = None
        torch.cuda.empty_cache()
    utils.clear_memory(clear_jax=False)

    repeats = int(os.environ.get("PHASE2B8_ABMPNN_REPEATS", "1"))
    attempts = []
    for attempt in range(1, repeats + 1):
        utils.clear_memory(clear_jax=False)
        started = time.perf_counter()
        sequences, success = redesign.run_abmpnn_redesign_pipeline(
            trajectory_pdb_af=str(trajectory_pdb_af),
            run_settings=run_settings,
            atom_distance_cutoff=run_settings["atom_distance_cutoff"],
            seed=seed,
        )
        attempts.append(
            {
                "attempt": attempt,
                "seconds": time.perf_counter() - started,
                "success": success,
                "sequence_count": len(sequences),
                "sequences": sequences,
            }
        )
    sequences = attempts[-1]["sequences"]
    success = attempts[-1]["success"]
    report = {
        "status": (
            "PASS"
            if all(
                attempt["success"]
                and attempt["sequence_count"]
                == run_settings["max_mpnn_sequences"]
                and all(
                    float(item["score"]) == float(item["score"])
                    for item in attempt["sequences"]
                )
                for attempt in attempts
            )
            else "FAIL"
        ),
        "seed": seed,
        "af2_seconds": af2_seconds,
        "abmpnn_repeats": repeats,
        "abmpnn_attempts": attempts,
        "af2_final_sequence": trajectory_sequence,
        "af2_boundary_pdb": str(boundary_source.relative_to(REPO_ROOT)),
        "af2_boundary_pdb_sha256": sha256(boundary_source),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "jax_backend": JAX.default_backend(),
        "jax_devices": [str(device) for device in JAX.devices()],
        "success": success,
        "sequence_count": len(sequences),
        "sequences": sequences,
    }
    output_path = output_root / "abmpnn_production_state_probe.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
