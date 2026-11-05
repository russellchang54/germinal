#!/usr/bin/env python3
"""Validate the AF2-parent -> AF3-artifact -> AbMPNN process boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax_plugins.mps
import numpy as np
import yaml
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from germinal.filters import redesign  # noqa: E402
from germinal.utils import config as config_module  # noqa: E402


EXPECTED_SEQUENCE_SHA256 = (
    "af247058856a4f3d1151e2a86472ebd4f9ab6f6994c68eef0077b94a6b3a9a40"
)
EXPECTED_AF3_BOUNDARY_SHA256 = (
    "50cf77a53c6c65213cc732956494a584"
    "88ea9cecfbc4fa650bf8c088da7e8e41"
)
AF3_PYTHON = REPO_ROOT / ".envs/af3-mps-probe/bin/python"
AF3_RUNNER = REPO_ROOT / "scripts/run_af3_mps.py"
AF3_SOURCE = REPO_ROOT / ".envs/af3-mps-probe/share/af3-src"
AF3_MODELS = REPO_ROOT / ".envs/af3-mps-probe/share/af3-models"
AF3_TOOLS = REPO_ROOT / ".envs/af3-mps-tools/bin"
AF3_PATCH = (
    DOCS / "patches/tokamax-0.0.12-mps-f32.patch"
)
AF3_REVISION = "8f8ac383afba995f5953182d39a560ffb8bbbfd3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_identity() -> dict:
    distribution = importlib.metadata.distribution("jax-mps")
    plugin_root = Path(jax_plugins.mps.__file__).resolve().parent
    return {
        "pid": os.getpid(),
        "executable": str(Path(sys.executable)),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "jax_mps_version": distribution.version,
        "jax_plugins_mps_path": str(plugin_root),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "PYTHONHASHSEED",
                "VIRTUAL_ENV",
            )
        },
    }


def validate_af3_artifacts(structures: Path) -> dict:
    input_json = structures / "af3_inputs/il3_nb_s1791095845.json"
    subprocess_log = (
        structures
        / "af3_inputs/il3_nb_s1791095845_af3_subprocess.log"
    )
    preserved_data = (
        structures
        / "af3_outputs_preserved/il3_nb_s1791095845/"
        "il3_nb_s1791095845_data.json"
    )
    af3_pdb = structures / "il3_nb_s1791095845_af3.pdb"
    relaxed_pdb = structures / "il3_nb_s1791095845_relaxed.pdb"
    input_payload = json.loads(input_json.read_text())
    data_payload = json.loads(preserved_data.read_text())
    log_text = subprocess_log.read_text(errors="replace")
    target = input_payload["sequences"][1]["protein"]
    binder = input_payload["sequences"][0]["protein"]
    database_names = (
        "bfd-first_non_consensus_sequences.fasta",
        "uniref90_2022_05.fa",
        "uniprot_all_2021_04.fa",
        "mgy_clusters_2022_05.fa",
    )
    data_sequences = data_payload["sequences"]
    data_by_chain = {
        item["protein"]["id"]: item["protein"]
        for item in data_sequences
    }
    return {
        "input_json": {
            "path": str(input_json.relative_to(REPO_ROOT)),
            "sha256": sha256(input_json),
        },
        "subprocess_log": {
            "path": str(subprocess_log.relative_to(REPO_ROOT)),
            "sha256": sha256(subprocess_log),
        },
        "preserved_data_json": {
            "path": str(preserved_data.relative_to(REPO_ROOT)),
            "sha256": sha256(preserved_data),
        },
        "af3_pdb": {
            "path": str(af3_pdb.relative_to(REPO_ROOT)),
            "sha256": sha256(af3_pdb),
        },
        "relaxed_pdb": {
            "path": str(relaxed_pdb.relative_to(REPO_ROOT)),
            "sha256": sha256(relaxed_pdb),
        },
        "binder_sequence": binder["sequence"],
        "checks": {
            "input_has_target_msa_path": (
                target.get("unpairedMsaPath") == "msas/target_A.a3m"
            ),
            "input_target_templates_empty": target.get("templates") == [],
            "binder_sequence_present": bool(binder.get("sequence")),
            "production_database_path_in_log": (
                "/Volumes/T9/database" in log_text
                and (
                    "--db_dir" in log_text
                    or "--db-dir" in log_text
                )
            ),
            "all_four_database_searches_completed": all(
                f"Finished Jackhmmer ({database_name})" in log_text
                for database_name in database_names
            ),
            "mps_device_selected": (
                "default_backend=mps" in log_text
                and "device=MPS:0" in log_text
            ),
            "binder_msa_consumed": bool(
                data_by_chain.get("B", {}).get("unpairedMsa")
            ),
            "target_msa_consumed": bool(
                data_by_chain.get("A", {}).get("unpairedMsa")
            ),
            "af3_pdb_present": af3_pdb.is_file(),
            "relaxed_pdb_present": relaxed_pdb.is_file(),
        },
    }


def run_af3_inference(
    af3_structures: Path,
    output_root: Path,
) -> dict:
    source_data = (
        af3_structures
        / "af3_outputs_preserved/il3_nb_s1791095845/"
        "il3_nb_s1791095845_data.json"
    )
    output_dir = output_root / "af3_inference_outputs"
    log_path = output_root / "af3_inference.log"
    command = [
        str(AF3_PYTHON),
        str(AF3_RUNNER),
        "--af3-source",
        str(AF3_SOURCE),
        "--expected-af3-revision",
        AF3_REVISION,
        "--model-dir",
        str(AF3_MODELS),
        "--tools-bin-dir",
        str(AF3_TOOLS),
        "--tokamax-patch",
        str(AF3_PATCH),
        "--output-dir",
        str(output_dir),
        "--json-path",
        str(source_data),
        "--num-diffusion-samples",
        "5",
        "--num-recycles",
        "10",
        "--timeout-seconds",
        "7200",
    ]
    started = time.perf_counter()
    with log_path.open("w") as log:
        log.write("COMMAND: " + json.dumps(command) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    text = log_path.read_text(errors="replace")
    sample_dirs = sorted(
        output_dir.glob("il3_nb_s1791095845/seed-*_sample-*")
    )
    ranking_file = (
        output_dir
        / "il3_nb_s1791095845/"
        "il3_nb_s1791095845_ranking_scores.csv"
    )
    return {
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": proc.returncode,
        "runtime_seconds": elapsed,
        "log": str(log_path.relative_to(REPO_ROOT)),
        "log_sha256": sha256(log_path),
        "source_data": str(source_data.relative_to(REPO_ROOT)),
        "source_data_sha256": sha256(source_data),
        "output_directory": str(output_dir.relative_to(REPO_ROOT)),
        "expected_three_seed_runs": len(
            set(
                re.findall(
                    r"Running model inference with seed (\d+)",
                    text,
                )
            )
        ),
        "sample_directory_count": len(sample_dirs),
        "ranking_scores_present": ranking_file.is_file(),
        "ranking_scores_sha256": (
            sha256(ranking_file) if ranking_file.is_file() else None
        ),
        "mps_selected": (
            "default_backend=mps" in text
            and "device=MPS:0" in text
        ),
        "inference_completed": "Done running 1 fold jobs." in text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--boundary-pdb",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--af3-structures",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--af2-trace",
        type=Path,
        required=True,
    )
    parser.add_argument("--run-af3", action="store_true")
    args = parser.parse_args()

    output_root = args.output.resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    boundary_pdb = args.boundary_pdb.resolve()
    af3_structures = args.af3_structures.resolve()
    af2_trace = json.loads(args.af2_trace.resolve().read_text())
    parent_runtime = runtime_identity()
    af3_artifacts = validate_af3_artifacts(af3_structures)
    af3_inference = (
        run_af3_inference(af3_structures, output_root)
        if args.run_af3
        else None
    )
    if sha256(boundary_pdb) != EXPECTED_AF3_BOUNDARY_SHA256:
        raise RuntimeError(
            "Unexpected AF3-AbMPNN boundary PDB hash: "
            f"{sha256(boundary_pdb)}"
        )

    overrides = [
        "run=vhh_il3_macos",
        "target=il3",
        "filter/initial=vhh_il3",
        "filter/final=vhh_il3",
    ]
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base=None,
    ):
        cfg = compose(config_name="config", overrides=overrides)
    processed = config_module.process_config(cfg)
    run_settings = processed["run"]
    target_settings = processed["target"]
    run_settings["project_dir"] = str(
        output_root / "orchestration_config"
    )
    _, run_settings = config_module.initialize_germinal_run(
        run_settings,
        target_settings,
    )
    run_settings["abmpnn_preserve_worker_logs"] = True

    records = []
    error = None
    try:
        for repeat in range(1, args.repeats + 1):
            worker_root = output_root / "workers" / f"repeat_{repeat:03d}"
            run_settings["abmpnn_worker_log_dir"] = str(worker_root)
            started = time.perf_counter()
            sequences, success = redesign.run_abmpnn_redesign_pipeline(
                trajectory_pdb_af=str(boundary_pdb),
                run_settings=run_settings,
                atom_distance_cutoff=run_settings["atom_distance_cutoff"],
                seed=1791095845,
            )
            elapsed = time.perf_counter() - started
            sequence_strings = [item["seq"] for item in sequences]
            scores = [float(item["score"]) for item in sequences]
            identity_paths = sorted(
                worker_root.glob("trial_*.runtime.json")
            )
            identities = [
                json.loads(path.read_text()) for path in identity_paths
            ]
            record = {
                "repeat": repeat,
                "success": success,
                "runtime_seconds": elapsed,
                "sequence_count": len(sequences),
                "unique_sequence_count": len(set(sequence_strings)),
                "all_scores_finite": bool(
                    scores and all(np.isfinite(score) for score in scores)
                ),
                "sequences_sha256": hashlib.sha256(
                    json.dumps(
                        sequence_strings, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "sequences_match_expected": (
                    hashlib.sha256(
                        json.dumps(
                            sequence_strings, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                    == EXPECTED_SEQUENCE_SHA256
                ),
                "worker_identities": identities,
                "worker_pid_unique": len(
                    {item["pid"] for item in identities}
                )
                == len(identities),
                "child_versions": [
                    item["jax_mps_version"] for item in identities
                ],
                "child_prefixes": [
                    item["sys_prefix"] for item in identities
                ],
                "child_backends": [
                    item["jax_default_backend"] for item in identities
                ],
            }
            (output_root / f"repeat_{repeat:03d}.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            records.append(record)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    checks = {
        "parent_runtime_is_af2_0_10_9": (
            parent_runtime["jax_mps_version"] == "0.10.9"
            and parent_runtime["sys_prefix"].endswith(
                ".envs/af2-mps-0109"
            )
            and parent_runtime["jax_default_backend"] == "mps"
        ),
        "af3_artifacts_valid": all(af3_artifacts["checks"].values()),
        "af3_inference_completed": (
            af3_inference is not None
            and af3_inference["status"] == "PASS"
            and af3_inference["expected_three_seed_runs"] == 3
            and af3_inference["sample_directory_count"] == 15
            and af3_inference["ranking_scores_present"]
            and af3_inference["mps_selected"]
            and af3_inference["inference_completed"]
        ),
        "af2_artifacts_present": (
            af2_trace["step_count"] >= 60
            and (REPO_ROOT / af2_trace["pdb"]["path"]).is_file()
        ),
        "af2_best_sequence_matches_af3_input": (
            af2_trace["best_sequence"]
            == af3_artifacts["binder_sequence"]
        ),
        "all_repeats_completed": len(records) == args.repeats,
        "all_repeats_returned_five": all(
            item["sequence_count"] == 5 for item in records
        ),
        "all_repeats_unique": all(
            item["unique_sequence_count"] == 5 for item in records
        ),
        "all_scores_finite": all(
            item["all_scores_finite"] for item in records
        ),
        "all_sequences_exact_expected": all(
            item["sequences_match_expected"] for item in records
        ),
        "all_children_used_0_10_11": all(
            item["child_versions"]
            and all(version == "0.10.11" for version in item["child_versions"])
            for item in records
        ),
        "all_children_used_abmpnn_environment": all(
            item["child_prefixes"]
            and all(
                prefix.endswith(".envs/abmpnn-mps-0111")
                for prefix in item["child_prefixes"]
            )
            for item in records
        ),
        "all_children_used_mps": all(
            item["child_backends"]
            and all(backend == "mps" for backend in item["child_backends"])
            for item in records
        ),
        "child_processes_distinct": (
            sum(
                len(item["worker_identities"]) for item in records
            )
            == len(
                {
                    identity["pid"]
                    for item in records
                    for identity in item["worker_identities"]
                }
            )
        ),
        "parent_and_child_versions_differ_as_required": (
            all(
                item["child_versions"]
                and all(
                    version != parent_runtime["jax_mps_version"]
                    for version in item["child_versions"]
                )
                for item in records
            )
        ),
    }
    report = {
        "phase": "2B.12",
        "stage": "component_boundary",
        "status": "PASS" if not error and all(checks.values()) else "FAIL",
        "error": error,
        "seed": 1791095845,
        "boundary_pdb": str(boundary_pdb.relative_to(REPO_ROOT)),
        "boundary_pdb_sha256": sha256(boundary_pdb),
        "parent_runtime": parent_runtime,
        "abmpnn_runtime_config": {
            "python": run_settings.get("abmpnn_python_path"),
            "expected_jax_mps_version": run_settings.get(
                "abmpnn_expected_jax_mps_version"
            ),
            "preserve_worker_logs": run_settings.get(
                "abmpnn_preserve_worker_logs"
            ),
        },
        "af3_artifacts": af3_artifacts,
        "af3_inference": af3_inference,
        "af2_boundary": {
            "trace_path": str(
                args.af2_trace.resolve().relative_to(REPO_ROOT)
            ),
            "best_sequence": af2_trace["best_sequence"],
            "final_sequence": af2_trace["final_sequence"],
            "pdb": af2_trace["pdb"],
            "step_count": af2_trace["step_count"],
        },
        "repeats": records,
        "checks": checks,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
