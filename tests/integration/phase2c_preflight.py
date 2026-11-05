#!/usr/bin/env python3
"""Freeze the Phase 2C inputs, runtimes, database state, and comparison policy."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2c"
DATABASE_ROOT = Path("/Volumes/T9/database")
SOURCE_FILES = (
    "run_germinal.py",
    "germinal/design/design.py",
    "germinal/filters/af3.py",
    "germinal/filters/filter_utils.py",
    "germinal/filters/pyrosetta_utils.py",
    "germinal/filters/redesign.py",
    "germinal/utils/utils.py",
    "scripts/af2_mps_compat.py",
    "scripts/run_af3_mps.py",
    "scripts/run_abmpnn_worker.py",
)
CONFIG_FILES = (
    "configs/config.yaml",
    "configs/run/vhh_il3.yaml",
    "configs/run/vhh_il3_macos.yaml",
    "configs/target/il3.yaml",
    "configs/filter/initial/vhh_il3.yaml",
    "configs/filter/final/vhh_il3.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def effective_configuration() -> dict:
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base=None,
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "run=vhh_il3_macos",
                "target=il3",
                "filter/initial=vhh_il3",
                "filter/final=vhh_il3",
                "project_dir=docs/porting/af3_mps_probe/results/"
                "phase2c/preflight_config_root",
                "pdb_dir=docs/porting/af3_mps_probe/results/"
                "phase2b4/run1/input",
                "+starting_pdb_complex=docs/porting/af3_mps_probe/"
                "results/phase2b4/run1/input/il3_11_8_18_nb.pdb",
                "af_params_dir=.envs/af2-mps-0109/share/"
                "af2-params/extracted",
                "dssp_path=.envs/germinal-macos/bin/mkdssp",
                "dalphaball_path=.envs/germinal-macos/bin/dalphaball",
                "af3_db_dir=/Volumes/T9/database",
                "pregenerate_seeds=true",
                "max_trajectories=1",
                "max_hallucinated_trajectories=1",
                "max_passing_designs=1",
            ],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def database_manifest() -> dict:
    readiness_path = DOCS / "phase2b6_database_manifest.json"
    readiness = json.loads(readiness_path.read_text())
    smoke_path = DOCS / "phase2b6_database_smoke.json"
    smoke = json.loads(smoke_path.read_text())
    required = readiness["expected_af3_database_manifest_after_provisioning"]
    files = {}
    for name, value in required.items():
        path = Path(value)
        files[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    status_path = DATABASE_ROOT / ".manifests/provisioning_status.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.is_file()
        else None
    )
    usage = shutil.disk_usage(DATABASE_ROOT)
    return {
        "phase": "2C",
        "database_root": str(DATABASE_ROOT),
        "database_ready": (
            readiness["status"] == "PASS_PRODUCTION_READY"
            and readiness["production_ready"] is True
            and status is not None
            and status.get("status") == "PASS_PRODUCTION_READY"
            and smoke.get("status") == "PASS"
            and all(item["exists"] for item in files.values())
        ),
        "readiness_manifest": {
            "path": str(readiness_path.relative_to(REPO_ROOT)),
            "sha256": sha256(readiness_path),
        },
        "live_status_manifest": (
            {
                "path": str(status_path),
                "sha256": sha256(status_path),
            }
            if status_path.is_file()
            else None
        ),
        "smoke_test": {
            "path": str(smoke_path.relative_to(REPO_ROOT)),
            "sha256": sha256(smoke_path),
            "status": smoke["status"],
            "af3_revision": smoke["af3_revision"],
            "backend": smoke["runtime"]["selected_backend"],
            "device": smoke["runtime"]["selected_device"],
        },
        "required_databases": files,
        "filesystem": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        "scope": (
            "Existing production-readiness and integrity evidence was verified "
            "against the current mount and required paths. Provisioning, "
            "replacement, and large-file checksum scans were not repeated."
        ),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    accepted = json.loads(
        (DOCS / "phase2b12_baseline_integrity.json").read_text()
    )
    provenance = json.loads(
        (DOCS / "phase2b12_backend_provenance.json").read_text()
    )
    release = json.loads(
        (DOCS / "phase2b12_release_decision.json").read_text()
    )
    runtime_probes = provenance["runtimes"]
    config = effective_configuration()
    database = database_manifest()
    source_hashes = {
        path: sha256(REPO_ROOT / path) for path in SOURCE_FILES
    }
    config_hashes = {
        path: sha256(REPO_ROOT / path) for path in CONFIG_FILES
    }
    input_dir = (
        DOCS / "results/phase2b4/run1/input"
    )
    inputs = {
        "starting_complex": {
            "path": str(
                (input_dir / "il3_11_8_18_nb.pdb").relative_to(REPO_ROOT)
            ),
            "sha256": sha256(input_dir / "il3_11_8_18_nb.pdb"),
        },
        "target": {
            "path": "pdbs/il3.pdb",
            "sha256": sha256(REPO_ROOT / "pdbs/il3.pdb"),
        },
        "binder": {
            "path": "pdbs/nb.pdb",
            "sha256": sha256(REPO_ROOT / "pdbs/nb.pdb"),
        },
    }
    checkpoints = accepted["checkpoints"]
    local_usage = shutil.disk_usage(REPO_ROOT)
    runtime_checks = {
        "af2": runtime_probes["af2"]["sys_prefix"].endswith(
            ".envs/af2-mps-0109"
        ),
        "af3": runtime_probes["af3"]["sys_prefix"].endswith(
            ".envs/af3-mps-probe"
        ),
        "abmpnn": runtime_probes["abmpnn"]["sys_prefix"].endswith(
            ".envs/abmpnn-mps-0111"
        ),
    }
    runtime_ok = (
        runtime_probes["af2"]["jax_mps_version"] == "0.10.9"
        and runtime_probes["af3"]["jax_mps_version"] == "0.10.9"
        and runtime_probes["abmpnn"]["jax_mps_version"] == "0.10.11"
        and all(runtime_checks.values())
    )
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    process_listing = subprocess.run(
        ["ps", "aux"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    active_component_lines = [
        line
        for line in process_listing.stdout.splitlines()
        if any(
            token in line
            for token in (
                "run_germinal.py",
                "run_abmpnn_worker.py",
                "run_af3_mps.py",
            )
        )
        and "phase2c_preflight.py" not in line
    ]
    preflight_checks = {
        "phase2b12_release_pass": release["status"] == "PASS",
        "git_commit_matches": current_commit == accepted["git_commit"],
        "runtime_split_correct": runtime_ok,
        "unchanged_source_hashes_match": accepted["checks"][
            "unchanged_source_hashes_match"
        ],
        "all_af2_checkpoints_match": all(
            item["matches"] for item in checkpoints
        ),
        "scientific_config_preserved": accepted["checks"][
            "scientific_configuration_matches_base"
        ],
        "production_database_ready": database["database_ready"],
        "no_active_component_process": (
            process_listing.returncode == 0
            and not active_component_lines
        ),
        "seed_exact": config["seed"] == 1791095845
        if "seed" in config
        else True,
        "af3_revision_exact": (
            config["af3_expected_revision"]
            == "8f8ac383afba995f5953182d39a560ffb8bbbfd3"
        ),
        "abmpnn_runtime_explicit": (
            config["abmpnn_python_path"]
            == ".envs/abmpnn-mps-0111/bin/python"
            and config["abmpnn_expected_jax_mps_version"] == "0.10.11"
        ),
    }
    comparison_policy = {
        "frozen_before_run1": True,
        "run_execution": "sequential",
        "required_complete_runs": 2,
        "normalization_allowed": [
            "run-specific absolute and relative output paths",
            "log timestamps and wall-clock timing fields",
            "design_time fields",
            "documented non-scientific CIF timestamps",
        ],
        "never_normalized": [
            "seed",
            "scientific configuration",
            "AF2 105 iteration records",
            "AF2 sequence logits, gradients, and model state",
            "AF2 best sequence, metrics, and PDB",
            "AbMPNN input PDB",
            "AbMPNN sequences, scores, and retry outcomes",
            "AF3 search completion and database-derived input content",
            "AF3 prediction and relaxed structures",
            "PyRosetta metrics",
            "filter decisions",
            "final candidate PDB hashes",
        ],
        "numeric_comparison": (
            "Exact equality. No post hoc tolerances will be introduced."
        ),
        "structural_comparison": (
            "PDB byte equality after allowed path/timestamp normalization. "
            "CIF metadata timestamps may be stripped; coordinates and "
            "confidence content remain exact."
        ),
        "failure_policy": (
            "If Run 1 fails any mandatory boundary, do not start Run 2. "
            "If Run 2 fails, report PARTIAL with the exact boundary evidence."
        ),
    }
    preflight = {
        "phase": "2C",
        "status": (
            "PASS" if all(preflight_checks.values()) else "FAIL"
        ),
        "checked_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "git_commit": current_commit,
        "git_status": subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": str(Path(sys.executable)),
            "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": os.sysconf("SC_PHYS_PAGES")
            * os.sysconf("SC_PAGE_SIZE"),
            "workspace_filesystem": {
                "total_bytes": local_usage.total,
                "free_bytes": local_usage.free,
            },
        },
        "active_component_process_check": {
            "command": "ps aux",
            "returncode": process_listing.returncode,
            "matching_processes": active_component_lines,
        },
        "checks": preflight_checks,
        "runtime_manifest": {
            "path": "docs/porting/af3_mps_probe/"
            "phase2c_runtime_manifest.json",
        },
        "database_manifest": {
            "path": "docs/porting/af3_mps_probe/"
            "phase2c_database_manifest.json",
        },
        "source_hashes": source_hashes,
        "config_hashes": config_hashes,
        "effective_config": config,
        "checkpoints": checkpoints,
        "inputs": inputs,
        "comparison_policy": comparison_policy,
    }
    runtime_manifest = {
        "phase": "2C",
        "source": "Phase 2B.12 runtime probe evidence",
        "runtimes": runtime_probes,
        "runtime_split": release["stage_scoped_runtime_decision"],
        "process_boundary": (
            "AF2 parent executes in .envs/af2-mps-0109; AF3 uses its "
            "configured dedicated interpreter; AbMPNN uses the explicit "
            ".envs/abmpnn-mps-0111 interpreter and emits a runtime identity."
        ),
        "status": "PASS" if runtime_ok else "FAIL",
    }
    write_json(DOCS / "phase2c_database_manifest.json", database)
    write_json(DOCS / "phase2c_runtime_manifest.json", runtime_manifest)
    write_json(DOCS / "phase2c_preflight.json", preflight)
    write_json(
        RESULTS / "frozen_effective_config.json",
        config,
    )
    print(json.dumps({"status": preflight["status"], "checks": preflight_checks}, indent=2))
    return 0 if preflight["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
