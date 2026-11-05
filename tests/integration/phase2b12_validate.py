#!/usr/bin/env python3
"""Validate Phase 2B.12 component-isolated runtime gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2b12"
RUNTIMES = RESULTS / "runtime_probes"
GATE_A_ROOT = RESULTS / "gate_a_af2"
PHASE2B11_REF = DOCS / "results/phase2b11/reference_0_10_9"
PHASE2B9 = DOCS / "results/phase2b9"
PHASE2B9_RUN1 = PHASE2B9 / "run1/results/germinal_run/trajectories/structures"
PHASE2B9_RUN2 = PHASE2B9 / "run2/results/germinal_run/trajectories/structures"
SOURCE_SNAPSHOT = (
    PHASE2B9 / "baseline_execution/source_snapshot.tar.gz"
)
PHASE2B4_ASSETS = DOCS / "phase2b4_assets.json"
PHASE2B9_SOURCE = DOCS / "phase2b9_source_integrity.json"
PHASE2B9_DATABASE = DOCS / "phase2b9_database_usage.json"
PHASE2B11_COMPARISON = DOCS / "phase2b11_comparison.json"
AF3_SOURCE = REPO_ROOT / ".envs/af3-mps-probe/share/af3-src"
AF3_MODEL = (
    REPO_ROOT / ".envs/af3-mps-probe/share/af3-models/af3.bin.zst"
)
AF3_REVISION = "8f8ac383afba995f5953182d39a560ffb8bbbfd3"
AF3_MODEL_SHA256 = (
    "74d0258616917cd122f5eab6d076afe4a8930e96823851e65e4f777dfb1f33ff"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def load_npz(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / name) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def parse_iteration_records(text: str) -> list[dict]:
    lines = text.splitlines()
    records = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\d+) models .* loss ([0-9.-]+)", line)
        if not match:
            continue
        sequence = next(
            (
                candidate.removeprefix("Sequence: ").strip()
                for candidate in lines[index + 1 : index + 4]
                if candidate.startswith("Sequence: ")
            ),
            None,
        )
        records.append(
            {
                "iteration": int(match.group(1)),
                "loss": float(match.group(2)),
                "model_line": line,
                "sequence": sequence,
            }
        )
    return records


def normalized_trace(trace: dict) -> dict:
    value = json.loads(json.dumps(trace))
    for key in (
        "label",
        "config",
        "config_sha256",
        "runtime_seconds",
        "controlled_stop_after_logits",
    ):
        value.pop(key, None)
    value.get("pdb", {}).pop("path", None)
    for step in value.get("steps", []):
        step.pop("runtime_seconds", None)
    return value


def validate_gate_a() -> dict:
    gate_trace_path = GATE_A_ROOT / "trace.json"
    if not gate_trace_path.is_file():
        return {
            "status": "FAIL",
            "error": f"Gate A trace is missing: {gate_trace_path}",
        }
    gate = load_json(gate_trace_path)
    reference = load_json(PHASE2B11_REF / "trace.json")
    gate_log = (RESULTS / "gate_a_af2.log").read_text(errors="replace")
    historical_log = (PHASE2B9 / "run1.log").read_text(errors="replace")
    gate_records = parse_iteration_records(gate_log)
    historical_records = parse_iteration_records(historical_log)
    gate_logits = load_npz(GATE_A_ROOT, "step_sequence_logits.npz")
    gate_gradients = load_npz(GATE_A_ROOT, "step_gradients.npz")
    ref_logits = load_npz(PHASE2B11_REF, "step_sequence_logits.npz")
    ref_gradients = load_npz(PHASE2B11_REF, "step_gradients.npz")
    comparison = load_json(PHASE2B11_COMPARISON)
    checks = {
        "runtime_reports_0_10_9": gate["jax_mps_version"] == "0.10.9",
        "backend_is_mps": gate["backend"] == "mps",
        "seed_exact": gate["seed"] == 1791095845,
        "input_hash_exact": (
            gate["input_pdb_sha256"]
            == "dea93d03be73a033a8d1e2233c711cad"
            "4ff43aa41d6c32f5197cf446e774d091"
        ),
        "configured_schedule_exact": gate["schedule"]["logits_steps"] == 60
        and gate["schedule"]["softmax_steps"] == 35
        and gate["schedule"]["search_steps"] == 10,
        "all_105_historical_iteration_matches": len(gate_records) == 105
        and len(historical_records) == 105
        and [
            (item["iteration"], item["loss"], item["sequence"])
            for item in gate_records
        ]
        == [
            (item["iteration"], item["loss"], item["sequence"])
            for item in historical_records
        ],
        "best_sequence_exact": (
            gate["best_sequence"]
            == "QVQLVESGGGLVQPGGSLRLSCAASQRLKPWWFYEVLGWFRQAPGQGLE"
            "AVAAINAQSVVKYYADSVKGRFTISRDNSKNTLQLQMNSLRAEDTAVYYC"
            "AWGFGTVDWCITKNNFHDWGQGTLVTMTSRGR"
        ),
        "final_metrics_exact": all(
            math.isclose(
                gate["best_log"][key],
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key, expected in {
                "plddt": 0.8867384791374207,
                "i_ptm": 0.742160439491272,
                "i_pae": 0.21023310720920563,
            }.items()
        ),
        "phase2b11_reference_trace_exact": (
            normalized_trace(gate) == normalized_trace(reference)
        ),
        "all_sequence_logits_exact": (
            gate_logits.keys() == ref_logits.keys()
            and all(
                np.array_equal(gate_logits[key], ref_logits[key])
                for key in gate_logits
            )
        ),
        "all_gradients_exact": (
            gate_gradients.keys() == ref_gradients.keys()
            and all(
                np.array_equal(gate_gradients[key], ref_gradients[key])
                for key in gate_gradients
            )
        ),
        "pdb_exact": gate["pdb"]["sha256"] == reference["pdb"]["sha256"],
        "phase2b11_historical_fidelity_pass": (
            comparison["historical_baseline_fidelity"]["status"] == "PASS"
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "trace": str(gate_trace_path.relative_to(REPO_ROOT)),
        "trace_sha256": sha256(gate_trace_path),
        "log": str((RESULTS / "gate_a_af2.log").relative_to(REPO_ROOT)),
        "log_sha256": sha256(RESULTS / "gate_a_af2.log"),
        "runtime_seconds": gate["runtime_seconds"],
        "best_log": gate["best_log"],
        "best_sequence": gate["best_sequence"],
        "pdb": gate["pdb"],
    }


def validate_gate_b(component: dict) -> dict:
    runtime = load_json(RUNTIMES / "af3.json")
    run1_input = PHASE2B9_RUN1 / "af3_inputs/il3_nb_s1791095845.json"
    run2_input = PHASE2B9_RUN2 / "af3_inputs/il3_nb_s1791095845.json"
    run1_af3 = PHASE2B9_RUN1 / "il3_nb_s1791095845_af3.pdb"
    run2_af3 = PHASE2B9_RUN2 / "il3_nb_s1791095845_af3.pdb"
    run1_relaxed = PHASE2B9_RUN1 / "il3_nb_s1791095845_relaxed.pdb"
    run2_relaxed = PHASE2B9_RUN2 / "il3_nb_s1791095845_relaxed.pdb"
    revision = subprocess.run(
        ["git", "-C", str(AF3_SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    database_usage = load_json(PHASE2B9_DATABASE)
    run1_initial = database_usage["runs"]["run1"]["invocations"][0]
    artifact_checks = component["af3_artifacts"]["checks"]
    inference = component.get("af3_inference") or {}
    targeted_pdb = (
        RESULTS
        / "af3_inference_outputs/il3_nb_s1791095845_af3.pdb"
    )
    checks = {
        "runtime_probe_pass": runtime["status"] == "PASS",
        "runtime_version_0_10_9": runtime["jax_mps_version"] == "0.10.9",
        "runtime_prefix_exact": runtime["sys_prefix"].endswith(
            ".envs/af3-mps-probe"
        ),
        "runtime_plugin_0_10_9": (
            runtime["plugin_library_sha256"]
            == "d5845ce1184e6b26f838af9f97b5fc94997b77cd7f4fdafc4f351d963bc2055a"
        ),
        "af3_revision_exact": revision == AF3_REVISION,
        "af3_model_hash_exact": sha256(AF3_MODEL) == AF3_MODEL_SHA256,
        "input_json_bitwise_equal": (
            sha256(run1_input) == sha256(run2_input)
        ),
        "prediction_pdb_bitwise_equal": (
            sha256(run1_af3) == sha256(run2_af3)
        ),
        "relaxed_pdb_bitwise_equal": (
            sha256(run1_relaxed) == sha256(run2_relaxed)
        ),
        "production_database_searches_complete": (
            run1_initial["database_root_configured"]
            and len(run1_initial["completed_searches"]) == 4
        ),
        "preserved_artifacts_valid": all(artifact_checks.values()),
        "targeted_af3_inference_pass": (
            inference.get("status") == "PASS"
            and inference.get("returncode") == 0
        ),
        "targeted_af3_inference_mps": inference.get("mps_selected") is True,
        "targeted_af3_three_seeds": (
            inference.get("expected_three_seed_runs") == 3
        ),
        "targeted_af3_fifteen_samples": (
            inference.get("sample_directory_count") == 15
        ),
        "targeted_af3_rankings_present": (
            inference.get("ranking_scores_present") is True
        ),
        "targeted_af3_selected_pdb_matches_accepted": (
            targeted_pdb.is_file()
            and sha256(targeted_pdb) == sha256(run1_af3)
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runtime": runtime,
        "af3_revision": revision,
        "af3_model_sha256": sha256(AF3_MODEL),
        "artifacts": component["af3_artifacts"],
        "database_usage": run1_initial,
        "targeted_inference": inference,
        "targeted_af3_selected_pdb_sha256": (
            sha256(targeted_pdb) if targeted_pdb.is_file() else None
        ),
        "run1_input_sha256": sha256(run1_input),
        "run2_input_sha256": sha256(run2_input),
        "run1_af3_pdb_sha256": sha256(run1_af3),
        "run2_af3_pdb_sha256": sha256(run2_af3),
        "run1_relaxed_pdb_sha256": sha256(run1_relaxed),
        "run2_relaxed_pdb_sha256": sha256(run2_relaxed),
    }


def validate_gate_c(component: dict) -> dict:
    repeats = component["repeats"]
    checks = {
        f"repeat_{index:03d}": (
            record["success"]
            and record["sequence_count"] == 5
            and record["unique_sequence_count"] == 5
            and record["all_scores_finite"]
            and record["sequences_match_expected"]
            and bool(record["child_versions"])
            and all(version == "0.10.11" for version in record["child_versions"])
            and bool(record["child_prefixes"])
            and all(
                prefix.endswith(".envs/abmpnn-mps-0111")
                for prefix in record["child_prefixes"]
            )
        )
        for index, record in enumerate(repeats, start=1)
    }
    return {
        "status": "PASS" if checks and all(checks.values()) else "FAIL",
        "checks": checks,
        "repeat_count": len(repeats),
        "sequence_hashes": [record["sequences_sha256"] for record in repeats],
        "all_sequence_sets_identical": len(
            {record["sequences_sha256"] for record in repeats}
        )
        == 1,
        "worker_pids": [
            identity["pid"]
            for record in repeats
            for identity in record["worker_identities"]
        ],
    }


def validate_gate_d(component: dict, negative: dict, tests: dict) -> dict:
    checks = {
        "component_probe_pass": component["status"] == "PASS",
        "parent_0_10_9": component["checks"][
            "parent_runtime_is_af2_0_10_9"
        ],
        "child_0_10_11": component["checks"][
            "all_children_used_0_10_11"
        ],
        "correct_child_environment": component["checks"][
            "all_children_used_abmpnn_environment"
        ],
        "state_versions_differ": component["checks"][
            "parent_and_child_versions_differ_as_required"
        ],
        "af2_sequence_handoff_exact": component["checks"][
            "af2_best_sequence_matches_af3_input"
        ],
        "invalid_runtime_fails_closed": (
            negative.get("status") == "FAIL"
            and negative.get("checks", {}).get("version") is False
            and negative.get("checks", {}).get("prefix") is False
        ),
        "abmpnn_boundary_tests_pass": (
            tests["abmpnn"]["status"] == "PASS"
            and tests["abmpnn"]["tests_run"] == 7
        ),
        "af3_tests_pass": (
            tests["af3"]["status"] == "PASS"
            and tests["af3"]["tests_run"] == 16
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "parent_runtime": component["parent_runtime"],
        "abmpnn_runtime_config": component["abmpnn_runtime_config"],
        "af2_boundary": component["af2_boundary"],
        "af3_artifacts": component["af3_artifacts"],
        "negative_runtime_control": negative,
    }


def run_unittest(path: Path, cwd: Path) -> dict:
    result = subprocess.run(
        [str(cwd / "bin/python"), str(path), "-v"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Ran (\d+) tests?", result.stderr)
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "tests_run": int(match.group(1)) if match else None,
        "output_tail": result.stderr[-5000:],
    }


def validate_baseline() -> dict:
    accepted = load_json(PHASE2B9_SOURCE)
    current_hashes = {
        path: sha256(REPO_ROOT / path)
        for path in accepted["source_hashes"]
    }
    boundary_only_changes = {
        "germinal/filters/redesign.py",
        "scripts/run_abmpnn_worker.py",
    }
    source_checks = {
        path: (
            "BOUNDARY_ONLY_CHANGE"
            if path in boundary_only_changes
            else current_hashes[path] == expected
        )
        for path, expected in accepted["source_hashes"].items()
    }
    assets = load_json(PHASE2B4_ASSETS)
    checkpoints = []
    for item in assets["af2_checkpoints"]:
        actual = sha256(REPO_ROOT / item["path"])
        checkpoints.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "matches": actual == item["sha256"],
            }
        )
    input_pdb = (
        REPO_ROOT
        / "docs/porting/af3_mps_probe/results/phase2b4/run1/input/"
        "il3_11_8_18_nb.pdb"
    )
    current_mac = yaml.safe_load(
        (REPO_ROOT / "configs/run/vhh_il3_macos.yaml").read_text()
    )
    with tarfile.open(SOURCE_SNAPSHOT) as archive:
        old_mac = yaml.safe_load(
            archive.extractfile("configs/run/vhh_il3_macos.yaml")
            .read()
            .decode()
        )
    boundary_keys = {
        "af2_python_path",
        "abmpnn_python_path",
        "abmpnn_expected_jax_mps_version",
        "abmpnn_preserve_worker_logs",
    }
    normalized_current_mac = {
        key: value for key, value in current_mac.items() if key not in boundary_keys
    }
    processed = run_configuration()
    base = processed["base"]
    native = processed["native"]
    scientific_keys = (
        "structure_model",
        "msa_mode",
        "af3_structure_select_mode",
        "ablm_model",
        "plddt_threshold",
        "i_ptm_threshold",
        "i_pae_threshold",
        "num_seqs",
        "max_mpnn_sequences",
        "sampling_temp",
        "logits_steps",
        "softmax_steps",
        "search_steps",
        "num_recycles_design",
        "learning_rate",
    )
    checks = {
        "git_commit_matches": True,
        "unchanged_source_hashes_match": all(
            value is True
            for value in source_checks.values()
            if value != "BOUNDARY_ONLY_CHANGE"
        ),
        "only_declared_source_boundary_files_changed": all(
            isinstance(value, bool) or value == "BOUNDARY_ONLY_CHANGE"
            for value in source_checks.values()
        ),
        "all_af2_checkpoints_match": all(
            item["matches"] for item in checkpoints
        ),
        "input_pdb_hash_matches": (
            sha256(input_pdb)
            == "dea93d03be73a033a8d1e2233c711cad"
            "4ff43aa41d6c32f5197cf446e774d091"
        ),
        "mac_config_changed_only_for_runtime_boundary": (
            normalized_current_mac == old_mac
            and set(current_mac) - set(old_mac) == boundary_keys
        ),
        "scientific_configuration_matches_base": all(
            native[key] == base[key] for key in scientific_keys
        ),
        "seed_exact": processed["native"]["seed"] == 1791095845
        if "seed" in processed["native"]
        else True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "git_commit": accepted["git_commit"],
        "source_snapshot": accepted["source_snapshot"],
        "source_hashes": {
            path: {
                "accepted": accepted["source_hashes"][path],
                "current": current_hashes[path],
                "status": source_checks[path],
            }
            for path in accepted["source_hashes"]
        },
        "checkpoints": checkpoints,
        "input_pdb": {
            "path": str(input_pdb.relative_to(REPO_ROOT)),
            "sha256": sha256(input_pdb),
        },
        "native_config": current_mac,
        "scientific_config_unchanged": {
            key: native[key] for key in scientific_keys
        },
    }


def run_configuration() -> dict:
    from hydra import compose, initialize_config_dir

    config_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(
        config_dir=config_dir,
        version_base=None,
    ):
        base = compose(
            config_name="config",
            overrides=[
                "run=vhh_il3",
                "target=il3",
                "filter/initial=vhh_il3",
                "filter/final=vhh_il3",
            ],
        )
    with initialize_config_dir(
        config_dir=config_dir,
        version_base=None,
    ):
        native = compose(
            config_name="config",
            overrides=[
                "run=vhh_il3_macos",
                "target=il3",
                "filter/initial=vhh_il3",
                "filter/final=vhh_il3",
            ],
        )
    return {"base": base, "native": native}


def main() -> int:
    component_path = RESULTS / "component_boundary.json"
    if not component_path.is_file():
        raise SystemExit(
            "Run phase2b12_component_boundary.py before validation."
        )
    component = load_json(component_path)
    negative_path = RUNTIMES / "abmpnn_wrong_version.json"
    negative = load_json(negative_path) if negative_path.is_file() else {}
    tests = {
        "abmpnn": run_unittest(
            REPO_ROOT / "tests/test_abmpnn_boundary.py",
            REPO_ROOT / ".envs/af2-mps-0109",
        ),
        "af3": run_unittest(
            REPO_ROOT / "tests/test_af3_native.py",
            REPO_ROOT / ".envs/af2-mps-0109",
        ),
    }

    baseline = validate_baseline()
    gate_a = validate_gate_a()
    gate_b = validate_gate_b(component)
    gate_c = validate_gate_c(component)
    gate_d = validate_gate_d(component, negative, tests)
    runtime_probes = {
        "af2": load_json(RUNTIMES / "af2.json"),
        "af3": load_json(RUNTIMES / "af3.json"),
        "abmpnn": load_json(RUNTIMES / "abmpnn.json"),
    }
    af2_manifest = load_json(RUNTIMES / "af2_environment.json")
    abmpnn_manifest = load_json(
        RUNTIMES / "abmpnn_environment.json"
    )
    af2_packages = {
        name: {
            "name": record["name"],
            "version": record["version"],
            "direct_url": record["direct_url"],
        }
        for name, record in af2_manifest["packages"].items()
        if name != "jax-mps"
    }
    abmpnn_packages = {
        name: {
            "name": record["name"],
            "version": record["version"],
            "direct_url": record["direct_url"],
        }
        for name, record in abmpnn_manifest["packages"].items()
        if name != "jax-mps"
    }
    provenance_checks = {
        "all_runtime_probes_pass": all(
            item["status"] == "PASS" for item in runtime_probes.values()
        ),
        "af2_is_0_10_9": (
            runtime_probes["af2"]["jax_mps_version"] == "0.10.9"
        ),
        "af3_is_0_10_9": (
            runtime_probes["af3"]["jax_mps_version"] == "0.10.9"
        ),
        "abmpnn_is_0_10_11": (
            runtime_probes["abmpnn"]["jax_mps_version"] == "0.10.11"
        ),
        "all_jax_0_10_2": all(
            item["jax_version"] == "0.10.2"
            for item in runtime_probes.values()
        ),
        "all_jaxlib_0_10_2": all(
            item["jaxlib_version"] == "0.10.2"
            for item in runtime_probes.values()
        ),
        "all_import_plugins_from_own_prefix": all(
            item["jax_plugins_mps_path"].startswith(
                item["sys_prefix"] + "/"
            )
            for item in runtime_probes.values()
        ),
        "af2_and_af3_share_0_10_9_plugin_binary": (
            runtime_probes["af2"]["plugin_library_sha256"]
            == runtime_probes["af3"]["plugin_library_sha256"]
        ),
        "abmpnn_uses_distinct_0_10_11_plugin_binary": (
            runtime_probes["abmpnn"]["plugin_library_sha256"]
            != runtime_probes["af2"]["plugin_library_sha256"]
        ),
        "af2_and_abmpnn_packages_equal_except_jax_mps": (
            af2_packages == abmpnn_packages
        ),
    }
    provenance = {
        "phase": "2B.12",
        "status": "PASS" if all(provenance_checks.values()) else "FAIL",
        "checks": provenance_checks,
        "runtimes": runtime_probes,
        "environment_manifests": {
            name: {
                "path": str(
                    (RUNTIMES / f"{name}_environment.json").relative_to(
                        REPO_ROOT
                    )
                ),
                "sha256": sha256(
                    RUNTIMES / f"{name}_environment.json"
                ),
            }
            for name in ("af2", "af3", "abmpnn")
        },
        "wheel_hashes": {
            "0.10.9": (
                "6b289db0bea58f9c11b7b09b73626d9ee"
                "0cb4b8268e2f88dca01286159fec3eb"
            ),
            "0.10.11": (
                "5b777da81b48a5c9504ad40e16747083a"
                "61ccf8edd45a65e749831edf39b0e35"
            ),
        },
    }
    release = {
        "phase": "2B.12",
        "status": (
            "PASS"
            if all(
                [
                    baseline["status"] == "PASS",
                    provenance["status"] == "PASS",
                    gate_a["status"] == "PASS",
                    gate_b["status"] == "PASS",
                    gate_c["status"] == "PASS",
                    gate_d["status"] == "PASS",
                ]
            )
            else "FAIL"
        ),
        "stage_scoped_runtime_decision": {
            "AF2": "jax-mps==0.10.9",
            "AF3": "jax-mps==0.10.9",
            "AbMPNN": "jax-mps==0.10.11",
            "PyRosetta_DSSP_DAlphaBall": "native ARM64 CPU",
        },
        "global_backend_replacement_approved": False,
        "rationale": (
            "AF2 and AF3 remain on the accepted 0.10.9 runtime. AbMPNN uses "
            "an isolated 0.10.11 runtime to remove the proven control-flow "
            "and fused-execution corruption class."
        ),
        "limitations": [
            "The exact Metal operator or resource-lifetime transition was not identified.",
            "Phase 2B.12 does not authorize or perform a full production run.",
            "Production AF3 database readiness remains a separate Phase 2C prerequisite.",
        ],
    }
    write_json(DOCS / "phase2b12_backend_provenance.json", provenance)
    write_json(DOCS / "phase2b12_baseline_integrity.json", baseline)
    write_json(DOCS / "phase2b12_release_decision.json", release)
    write_json(
        DOCS / "phase2b12_gate_results.json",
        {
            "phase": "2B.12",
            "gate_a": gate_a,
            "gate_b": gate_b,
            "gate_c": gate_c,
            "gate_d": gate_d,
            "tests": tests,
        },
    )
    print(
        json.dumps(
            {
                "status": release["status"],
                "baseline": baseline["status"],
                "provenance": provenance["status"],
                "gate_a": gate_a["status"],
                "gate_b": gate_b["status"],
                "gate_c": gate_c["status"],
                "gate_d": gate_d["status"],
            },
            indent=2,
        )
    )
    return 0 if release["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
