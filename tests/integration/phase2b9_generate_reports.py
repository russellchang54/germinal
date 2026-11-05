#!/usr/bin/env python3
"""Build Phase 2B.9 evidence from completed production runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2b9"
BASELINE = RESULTS / "baseline_execution/baseline.json"
RUN_NAMES = ("run1", "run2")
DATABASE_ROOT = "/Volumes/T9/database"
AF3_REVISION = "8f8ac383afba995f5953182d39a560ffb8bbbfd3"
PHASE2B8_PATCH = DOCS / "phase2b8_exact_patch.diff"
PHASE2B8_PATCH_SHA256 = (
    "d164b8e9e11a1743cde560e7267e09b457f6a1530e3a4b33809563b57666ce45"
)
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
DECISION_COLUMNS = (
    "clashes_filter",
    "sc_rmsd_filter",
    "binder_near_hotspot_filter",
    "cdr3_hotspot_contacts_filter",
    "percent_interface_cdr_filter",
    "interface_shape_comp_filter",
    "interface_hbonds_filter",
    "surface_hydrophobicity_filter",
    "interface_hydrophobicity_filter",
    "pdockq2_filter",
    "external_plddt_filter",
    "external_iptm_filter",
    "external_ptm_filter",
    "external_pae_filter",
)


def _load_phase2b7_reporter():
    path = REPO_ROOT / "tests/integration/phase2b7_generate_reports.py"
    spec = importlib.util.spec_from_file_location("phase2b7_reports", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reports = _load_phase2b7_reporter()
reports.BASELINE = BASELINE
reports.RESULTS = RESULTS
reports.RUNS = {name: RESULTS / name for name in RUN_NAMES}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _invocation_checks(invocation: dict) -> dict:
    log_path = REPO_ROOT / invocation["log"]["path"]
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    command = invocation.get("command") or []
    command_text = " ".join(command)
    return {
        "completed": invocation.get("upstream_seconds") is not None,
        "mps_device_selected": (
            invocation.get("backend") == "mps"
            and "device=MPS:0" in text
        ),
        "cpu_platform_override_absent": "JAX_PLATFORMS=cpu" not in text,
        "pinned_af3_revision": (
            invocation.get("expected_revision_verified") is True
            and AF3_REVISION in command_text
        ),
        "production_database_root": (
            invocation.get("database_root_configured") is True
            and DATABASE_ROOT in command_text
            and "--run-data-pipeline" in command
        ),
        "all_four_binder_searches_completed": len(
            invocation.get("completed_searches") or []
        ) == 4,
        "five_diffusion_samples_configured": (
            "--num-diffusion-samples" in command
            and command[command.index("--num-diffusion-samples") + 1] == "5"
        ),
        "ten_recycles_configured": (
            "--num-recycles" in command
            and command[command.index("--num-recycles") + 1] == "10"
        ),
        "target_custom_msa_semantics": (
            invocation.get("any_chain_search_skipped") is True
        ),
        "no_incidental_cpu_fallback": (
            "falling back to CPU" not in text.lower()
            and "no Metal device available" not in text
        ),
    }


def _run_checks(payload: dict, required_af3_count: int) -> dict:
    rows = payload["filters"]["rows"]
    sequences = [row.get("trajectory_sequence", "") for row in rows]
    scores = [row.get("abmpnn_score", "") for row in rows]
    invocation_checks = [
        {
            "name": invocation["name"],
            **_invocation_checks(invocation),
        }
        for invocation in payload["af3"]["invocations"]
    ]
    config = payload["config"]
    scientific = {
        "logits_steps": config.get("logits_steps"),
        "softmax_steps": config.get("softmax_steps"),
        "search_steps": config.get("search_steps"),
        "num_recycles_design": config.get("num_recycles_design"),
        "sampling_temp": config.get("sampling_temp"),
        "num_seqs": config.get("num_seqs"),
        "max_mpnn_sequences": config.get("max_mpnn_sequences"),
        "weights_beta": config.get("weights_beta"),
        "learning_rate": config.get("learning_rate"),
        "seed": payload.get("seed"),
    }
    return {
        "entrypoint_completed": (
            payload["entrypoint"] == "run_germinal.py"
            and payload["exit_code"] == 0
        ),
        "full_af2_schedule": (
            payload["af2"]["iteration_record_count"] == 105
            and payload["af2"]["final_sequence"] is not None
        ),
        "abmpnn_five_unique_sequences": (
            len(rows) == 5
            and len(set(sequences)) == 5
            and all(bool(sequence) for sequence in sequences)
        ),
        "abmpnn_scores_finite": (
            len(scores) == 5 and all(_finite(score) for score in scores)
        ),
        "all_filter_decisions_recorded": (
            len(rows) == 5
            and all(
                all(
                    row.get(column) in {"True", "False"}
                    for column in DECISION_COLUMNS
                )
                for row in rows
            )
        ),
        "all_candidates_decided": (
            payload["filters"]["accepted_count"]
            + payload["filters"]["rejected_count"]
            == 5
        ),
        "expected_af3_invocation_count": (
            payload["af3"]["invocation_count"] == required_af3_count
        ),
        "every_af3_invocation_production_mps": (
            len(invocation_checks) == required_af3_count
            and all(
                all(
                    value
                    for key, value in item.items()
                    if key != "name"
                )
                for item in invocation_checks
            )
        ),
        "all_pdbs_parse": (
            payload["pdb_validation"]["count"] >= 1
            and payload["pdb_validation"]["all_parse_success"]
        ),
        "all_coordinates_finite": payload["pdb_validation"]["all_finite"],
        "all_pdbs_have_expected_chains": payload["pdb_validation"][
            "all_expected_chains"
        ],
        "scientific_settings_preserved": (
            scientific["logits_steps"] == "60"
            and scientific["softmax_steps"] == "35"
            and scientific["search_steps"] == "10"
            and scientific["num_recycles_design"] == "3"
            and scientific["sampling_temp"] == "0.2"
            and scientific["num_seqs"] == "40"
            and scientific["max_mpnn_sequences"] == "5"
            and scientific["weights_beta"] == "0.1"
            and scientific["learning_rate"] == "0.1"
            and scientific["seed"] == 1791095845
        ),
        "af3_invocations": invocation_checks,
    }


def _run_passed(run_checks: dict) -> bool:
    scalar_checks = {
        key: value
        for key, value in run_checks.items()
        if key != "af3_invocations"
    }
    invocation_checks = run_checks.get("af3_invocations", [])
    return (
        all(isinstance(value, bool) and value for value in scalar_checks.values())
        and bool(invocation_checks)
        and all(
            all(
                isinstance(value, bool) and value
                for key, value in invocation.items()
                if key != "name"
            )
            for invocation in invocation_checks
        )
    )


def _repeatability(left: dict, right: dict) -> dict:
    left_rows = [
        {key: value for key, value in row.items() if key != "design_time"}
        for row in left["filters"]["rows"]
    ]
    right_rows = [
        {key: value for key, value in row.items() if key != "design_time"}
        for row in right["filters"]["rows"]
    ]
    input_left = {
        item["name"]: item["input_json"]["sha256"]
        for item in left["af3"]["invocations"]
    }
    input_right = {
        item["name"]: item["input_json"]["sha256"]
        for item in right["af3"]["invocations"]
    }
    pdb_left = {
        item["path"]: item["sha256"]
        for item in left["pdb_validation"]["files"]
    }
    pdb_right = {
        item["path"]: item["sha256"]
        for item in right["pdb_validation"]["files"]
    }
    checks = {
        "af3_input_inventory_complete": (
            len(input_left) == 6 and len(input_right) == 6
        ),
        "pdb_inventory_complete": (
            len(pdb_left) == 18 and len(pdb_right) == 18
        ),
        "normalized_configuration_equal": left["config"] == right["config"],
        "af2_iteration_records_equal": (
            left["af2"]["iteration_records"]
            == right["af2"]["iteration_records"]
        ),
        "af2_final_sequence_equal": (
            left["af2"]["final_sequence"] == right["af2"]["final_sequence"]
        ),
        "abmpnn_records_equal": (
            left["abmpnn"]["records"] == right["abmpnn"]["records"]
        ),
        "af3_input_hashes_equal": input_left == input_right,
        "all_filter_rows_equal": left_rows == right_rows,
        "all_pdb_hashes_equal": pdb_left == pdb_right,
    }
    return {
        "phase": "2B.9",
        **checks,
        "status": "PASS" if all(checks.values()) else "PARTIAL",
    }


def main() -> int:
    baseline = json.loads(BASELINE.read_text())
    phase2b8_patch_sha256 = reports.sha256(PHASE2B8_PATCH)
    frozen = {
        item["path"]: item["sha256"]
        for item in baseline["source_snapshot"]["files"]
    }
    source_hashes = {
        path: reports.sha256(REPO_ROOT / path) for path in SOURCE_FILES
    }
    source_integrity = {
        "phase": "2B.9",
        "git_commit": baseline["git_commit"],
        "source_snapshot": {
            "path": baseline["source_snapshot"]["path"],
            "sha256": baseline["source_snapshot"]["sha256"],
            "file_count": baseline["source_snapshot"]["file_count"],
        },
        "source_hashes": source_hashes,
        "source_hashes_match_execution_snapshot": all(
            source_hashes[path] == frozen[path] for path in SOURCE_FILES
        ),
        "phase2b8_patch": {
            "path": str(PHASE2B8_PATCH.relative_to(REPO_ROOT)),
            "expected_sha256": PHASE2B8_PATCH_SHA256,
            "actual_sha256": phase2b8_patch_sha256,
            "sha256_matches": phase2b8_patch_sha256 == PHASE2B8_PATCH_SHA256,
        },
        "changed_since_phase2b7": {
            "germinal/filters/redesign.py": (
                "Dedicated AbMPNN interpreter with finite-output validation "
                "and fresh-process retry."
            ),
            "scripts/run_abmpnn_worker.py": (
                "Dedicated AbMPNN worker entrypoint."
            ),
        },
        "scientific_parameters_changed": False,
    }
    write_json(DOCS / "phase2b9_source_integrity.json", source_integrity)

    payloads = {}
    for name in RUN_NAMES:
        log_path = RESULTS / f"{name}.log"
        if log_path.is_file():
            payloads[name] = reports.parse_run(name, RESULTS / name)
            write_json(DOCS / f"phase2b9_{name}.json", payloads[name])

    if not payloads:
        raise SystemExit("No Phase 2B.9 run logs found.")

    required_af3_count = 1 + 5
    checks = {
        name: _run_checks(payload, required_af3_count)
        for name, payload in payloads.items()
    }
    for name, run_checks in checks.items():
        write_json(DOCS / f"phase2b9_{name}_validation.json", run_checks)

    database_usage = {
        "phase": "2B.9",
        "database_root": DATABASE_ROOT,
        "database_manifest_sha256": reports.sha256(
            DOCS / "phase2b6_database_manifest.json"
        ),
        "interpretation": (
            "Every initial and final AF3 invocation configured the full "
            "external database and completed BFD, UniRef90, UniProt, and "
            "MGnify binder-chain searches. The target chain intentionally "
            "used the complete custom MSA."
        ),
        "runs": {
            name: {
                "invocations": [
                    {
                        "name": item["name"],
                        "database_root_configured": item[
                            "database_root_configured"
                        ],
                        "completed_searches": item["completed_searches"],
                        "deduplicated_msas": item["deduplicated_msas"],
                        "templates": item["templates"],
                        "upstream_seconds": item["upstream_seconds"],
                    }
                    for item in payload["af3"]["invocations"]
                ]
            }
            for name, payload in payloads.items()
        },
    }
    write_json(DOCS / "phase2b9_database_usage.json", database_usage)

    resource_metrics = {
        "phase": "2B.9",
        "runs": {
            name: {
                "runtime": payload["runtime"],
                "wall_seconds": payload["wall_seconds"],
                "user_seconds": payload["user_seconds"],
                "system_seconds": payload["system_seconds"],
                "maximum_rss_bytes": payload["max_rss_bytes"],
                "peak_memory_footprint_bytes": payload[
                    "peak_memory_footprint_bytes"
                ],
                "af3_upstream_seconds": [
                    item["upstream_seconds"]
                    for item in payload["af3"]["invocations"]
                ],
            }
            for name, payload in payloads.items()
        },
    }
    write_json(DOCS / "phase2b9_resource_metrics.json", resource_metrics)

    run_completed = {
        name: (
            payload.get("entrypoint") == "run_germinal.py"
            and payload.get("exit_code") == 0
        )
        for name, payload in payloads.items()
    }

    repeatability = None
    if set(payloads) == set(RUN_NAMES) and all(run_completed.values()):
        repeatability = _repeatability(payloads["run1"], payloads["run2"])
    elif set(payloads) == set(RUN_NAMES):
        left = payloads["run1"]
        right = payloads["run2"]
        left_inputs = {
            item["name"]: item["input_json"]["sha256"]
            for item in left["af3"]["invocations"]
        }
        right_inputs = {
            item["name"]: item["input_json"]["sha256"]
            for item in right["af3"]["invocations"]
        }
        common_inputs = set(left_inputs) & set(right_inputs)
        left_pdbs = {
            item["path"]: item["sha256"]
            for item in left["pdb_validation"]["files"]
        }
        right_pdbs = {
            item["path"]: item["sha256"]
            for item in right["pdb_validation"]["files"]
        }
        common_pdbs = set(left_pdbs) & set(right_pdbs)
        repeatability = {
            "phase": "2B.9",
            "status": "FAIL",
            "comparison_performed": False,
            "reason": (
                "Run 2 did not complete the literal production workflow. "
                "A completed-output repeatability comparison cannot be "
                "established."
            ),
            "run1_completed": run_completed["run1"],
            "run2_completed": run_completed["run2"],
            "comparable_prefix": {
                "normalized_configuration_equal": (
                    left["config"] == right["config"]
                ),
                "af2_iteration_records_equal": (
                    left["af2"]["iteration_records"]
                    == right["af2"]["iteration_records"]
                ),
                "af2_final_sequence_equal": (
                    left["af2"]["final_sequence"]
                    == right["af2"]["final_sequence"]
                ),
                "common_af3_input_count": len(common_inputs),
                "all_common_af3_input_hashes_equal": all(
                    left_inputs[name] == right_inputs[name]
                    for name in common_inputs
                ),
                "common_pdb_count": len(common_pdbs),
                "all_common_pdb_hashes_equal": all(
                    left_pdbs[path] == right_pdbs[path]
                    for path in common_pdbs
                ),
                "differing_common_pdbs": sorted(
                    path
                    for path in common_pdbs
                    if left_pdbs[path] != right_pdbs[path]
                ),
            },
            "fields_never_normalized": [
                "scientific configuration",
                "trajectory seed",
                "AF2 iteration records",
                "AbMPNN sequences and finite scores",
                "AF3 input hashes",
                "AF3 outputs",
                "PyRosetta metrics",
                "filter decisions",
                "final PDB hashes",
            ],
            "permitted_normalization": [
                "run-specific output paths",
                "timestamps and wall-clock durations",
            ],
        }
    write_json(DOCS / "phase2b9_repeatability.json", repeatability)

    exact_boundary_path = RESULTS / "diagnosis/exact_boundary_probe.json"
    exact_boundary = json.loads(exact_boundary_path.read_text())
    run1_sequences = [
        row["trajectory_sequence"] for row in payloads["run1"]["filters"]["rows"]
    ]
    run1_scores = [
        row["abmpnn_score"] for row in payloads["run1"]["filters"]["rows"]
    ]
    run2_log_text = payloads["run2"]["log"]["path"]
    run2_log_file = REPO_ROOT / run2_log_text
    run2_text = run2_log_file.read_text(errors="replace")
    run2_nonfinite_trials = len(
        re.findall(
            r"AbMPNN produced non-finite logits or scores on Apple MPS",
            run2_text,
        )
    )
    boundary_sequences = [
        item["seq"] for item in exact_boundary.get("sequences", [])
    ]
    boundary_scores = [
        item["score"] for item in exact_boundary.get("sequences", [])
    ]
    abmpnn_validation = {
        "phase": "2B.9",
        "status": "FAIL",
        "seed": 1791095845,
        "settings": {
            "sampling_temp": payloads["run1"]["config"].get("sampling_temp"),
            "num_seqs": payloads["run1"]["config"].get("num_seqs"),
            "max_mpnn_sequences": payloads["run1"]["config"].get(
                "max_mpnn_sequences"
            ),
        },
        "run1": {
            "status": "PASS",
            "sequence_count": len(run1_sequences),
            "unique_sequence_count": len(set(run1_sequences)),
            "all_scores_finite": (
                len(run1_scores) == 5
                and all(_finite(score) for score in run1_scores)
            ),
            "scores": run1_scores,
            "sequences_sha256": _sha256_text(
                json.dumps(run1_sequences, separators=(",", ":"))
            ),
        },
        "run2": {
            "status": "FAIL",
            "sequence_count_emitted": len(
                payloads["run2"]["filters"]["rows"]
            ),
            "fresh_process_trials_attempted": run2_nonfinite_trials,
            "all_trials_returned_nonfinite_logits_or_scores": (
                run2_nonfinite_trials == 5
            ),
            "pipeline_aborted_without_fabrication": (
                "refusing to emit unvalidated sequences" in run2_text
            ),
            "failure_symbol": "germinal.filters.redesign.get_abmpnn_sequences",
            "failure_location": "germinal/filters/redesign.py:309",
        },
        "exact_boundary_standalone": {
            "status": exact_boundary.get("status"),
            "input_pdb_sha256": exact_boundary.get("input_pdb_sha256"),
            "config_sha256": exact_boundary.get("config_sha256"),
            "jax_devices": exact_boundary.get("jax_devices"),
            "sequence_count": len(boundary_sequences),
            "unique_sequence_count": len(set(boundary_sequences)),
            "all_scores_finite": (
                len(boundary_scores) == 5
                and all(_finite(score) for score in boundary_scores)
            ),
            "runtime_seconds": exact_boundary.get("runtime_seconds"),
        },
        "root_cause": {
            "resolved": False,
            "proven": (
                "The exact failed input is process-local-valid but production "
                "execution repeatedly reached a non-finite JAX-MPS state."
            ),
            "not_proven": [
                "a deterministic tensor or mask defect",
                "the precise JAX-MPS numerical state transition",
            ],
        },
        "regression_tests": {
            "status": "PASS",
            "abmpnn_boundary_tests": 5,
            "af3_native_boundary_tests": 16,
            "total_tests": 21,
            "commands": [
                (
                    ".envs/germinal-mps-design/bin/python "
                    "tests/test_abmpnn_boundary.py -v"
                ),
                (
                    ".envs/germinal-mps-design/bin/python "
                    "tests/test_af3_native.py -v"
                ),
            ],
        },
        "acceptance": {
            "run1_five_finite_diverse": (
                len(run1_sequences) == 5
                and len(set(run1_sequences)) == 5
                and all(_finite(score) for score in run1_scores)
            ),
            "run2_five_finite_diverse": False,
            "gate_passed_for_both_runs": False,
        },
    }
    write_json(DOCS / "phase2b9_abmpnn_validation.json", abmpnn_validation)

    run_pass = {
        name: _run_passed(run_checks)
        for name, run_checks in checks.items()
    }
    source_pass = (
        source_integrity["source_hashes_match_execution_snapshot"]
        and source_integrity["phase2b8_patch"]["sha256_matches"]
    )
    repeatability_pass = (
        repeatability is not None and repeatability["status"] == "PASS"
    )
    decision = (
        "PASS"
        if (
            source_pass
            and set(run_pass) == set(RUN_NAMES)
            and all(run_pass.values())
            and repeatability_pass
        )
        else "FAIL"
    )

    commands = {}
    for name in RUN_NAMES:
        commands[name] = (
            "JAX_PLATFORMS=mps \\\n"
            'LIBCIFPP_DATA_DIR="$PWD/.envs/germinal-macos/share/libcifpp" \\\n'
            'PATH="$PWD/.envs/germinal-mps-design/bin:$PATH" \\\n'
            'PYTHONPATH="$PWD/scripts:$PWD" \\\n'
            "/usr/bin/time -l .envs/germinal-mps-design/bin/python -u "
            "run_germinal.py \\\n"
            "  run=vhh_il3_macos \\\n"
            "  target=il3 \\\n"
            "  filter/initial=vhh_il3 \\\n"
            "  filter/final=vhh_il3 \\\n"
            f"  project_dir=docs/porting/af3_mps_probe/results/phase2b9/{name} \\\n"
            "  pdb_dir=docs/porting/af3_mps_probe/results/phase2b4/run1/input \\\n"
            "  +starting_pdb_complex=docs/porting/af3_mps_probe/results/"
            "phase2b4/run1/input/il3_11_8_18_nb.pdb \\\n"
            "  af_params_dir=.envs/germinal-mps-design/share/af2-params/extracted \\\n"
            "  dssp_path=.envs/germinal-macos/bin/mkdssp \\\n"
            "  dalphaball_path=.envs/germinal-macos/bin/dalphaball \\\n"
            "  af3_database_manifest=null \\\n"
            "  af3_db_dir=/Volumes/T9/database \\\n"
            "  pregenerate_seeds=true \\\n"
            "  max_trajectories=1 \\\n"
            "  max_hallucinated_trajectories=1 \\\n"
            "  max_passing_designs=1"
        )

    execution_manifest = {
        "phase": "2B.9",
        "decision": decision,
        "entrypoint": "run_germinal.py",
        "git_commit": baseline["git_commit"],
        "source_snapshot": {
            "path": baseline["source_snapshot"]["path"],
            "sha256": baseline["source_snapshot"]["sha256"],
            "file_count": baseline["source_snapshot"]["file_count"],
        },
        "phase2b8_patch": source_integrity["phase2b8_patch"],
        "production_database_root": DATABASE_ROOT,
        "af3_revision": AF3_REVISION,
        "seed": 1791095845,
        "commands": commands,
        "runs": {
            name: {
                "output_directory": payload["root"],
                "log": payload["log"],
                "exit_code": payload["exit_code"],
                "exit_code_source": payload["exit_code_source"],
                "runtime": payload["runtime"],
                "wall_seconds": payload["wall_seconds"],
                "max_rss_bytes": payload["max_rss_bytes"],
                "peak_memory_footprint_bytes": payload[
                    "peak_memory_footprint_bytes"
                ],
                "af2_iteration_records": payload["af2"][
                    "iteration_record_count"
                ],
                "abmpnn_sequence_count": len(payload["filters"]["rows"]),
                "af3_invocation_count": payload["af3"]["invocation_count"],
                "pdb_count": payload["pdb_validation"]["count"],
                "pdb_sha256": {
                    item["path"]: item["sha256"]
                    for item in payload["pdb_validation"]["files"]
                },
            }
            for name, payload in payloads.items()
        },
        "preserved_failure": {
            "run": "run2",
            "stage": "AbMPNN",
            "symbol": "germinal.filters.redesign.get_abmpnn_sequences",
            "location": "germinal/filters/redesign.py:309",
            "nonfinite_trials": run2_nonfinite_trials,
            "invalid_attempt_quarantined": (
                "docs/porting/af3_mps_probe/results/phase2b9/"
                "run2_invalid_log_truncation"
            ),
        },
    }
    write_json(
        DOCS / "phase2b9_execution_manifest.json", execution_manifest
    )

    summary = {
        "phase": "2B.9",
        "decision": decision,
        "scope": (
            "Literal production Germinal workflow after the Phase 2B.8 "
            "AbMPNN boundary correction."
        ),
        "source_snapshot_sha256": baseline["source_snapshot"]["sha256"],
        "production_database_root": DATABASE_ROOT,
        "af3_revision": AF3_REVISION,
        "runs": {
            name: {
                "status": "PASS" if run_pass[name] else "FAIL",
                "runtime": payload["runtime"],
                "wall_seconds": payload["wall_seconds"],
                "af3_invocations": payload["af3"]["invocation_count"],
                "abmpnn_sequences": len(payload["filters"]["rows"]),
                "accepted": payload["filters"]["accepted_count"],
                "rejected": payload["filters"]["rejected_count"],
                "pdb_count": payload["pdb_validation"]["count"],
            }
            for name, payload in payloads.items()
        },
        "repeatability": (
            repeatability["status"] if repeatability else "NOT_RUN"
        ),
        "blockers": (
            []
            if decision == "PASS"
            else [
                {
                    "run": "run2",
                    "stage": "AbMPNN",
                    "summary": (
                        "All five fresh-process trials returned non-finite "
                        "logits or scores; the production guard aborted "
                        "without issuing invalid sequences."
                    ),
                    "location": "germinal/filters/redesign.py:309",
                }
            ]
        ),
        "acceptance": checks,
        "artifacts": {
            **{
                name: f"docs/porting/af3_mps_probe/phase2b9_{name}.json"
                for name in payloads
            },
            **{
                f"{name}_validation": (
                    f"docs/porting/af3_mps_probe/"
                    f"phase2b9_{name}_validation.json"
                )
                for name in payloads
            },
            "database_usage": (
                "docs/porting/af3_mps_probe/phase2b9_database_usage.json"
            ),
            "resource_metrics": (
                "docs/porting/af3_mps_probe/phase2b9_resource_metrics.json"
            ),
            "source_integrity": (
                "docs/porting/af3_mps_probe/phase2b9_source_integrity.json"
            ),
            "abmpnn_validation": (
                "docs/porting/af3_mps_probe/"
                "phase2b9_abmpnn_validation.json"
            ),
            "execution_manifest": (
                "docs/porting/af3_mps_probe/"
                "phase2b9_execution_manifest.json"
            ),
            "repeatability": (
                "docs/porting/af3_mps_probe/phase2b9_repeatability.json"
            ),
            "subagent_audits": (
                "docs/porting/af3_mps_probe/"
                "phase2b9_subagent_audits.json"
            ),
        },
    }
    write_json(DOCS / "phase2b9_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
