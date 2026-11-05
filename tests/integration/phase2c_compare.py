#!/usr/bin/env python3
"""Compare the two complete Phase 2C production runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2c"
RUN_NAMES = ("run1", "run2")
EXPECTED_SEQUENCE_SHA256 = (
    "af247058856a4f3d1151e2a86472ebd4f9ab6f6994c68eef0077b94a6b3a9a40"
)
AF2_SEQUENCE = (
    "QVQLVESGGGLVQPGGSLRLSCAASQRLKPWWFYEVLGWFRQAPGQGLEAVAAINAQSVV"
    "KYYADSVKGRFTISRDNSKNTLQLQMNSLRAEDTAVYYCAWGFGTVDWCITKNNFHDWGQG"
    "TLVTMTSRGR"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_reporter():
    path = REPO_ROOT / "tests/integration/phase2b7_generate_reports.py"
    spec = importlib.util.spec_from_file_location("phase2c_reporter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalized_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in {"project_dir"}
    }


def normalized_filter_rows(payload: dict) -> list[dict]:
    ignored = {"design_time"}
    return [
        {
            key: value
            for key, value in row.items()
            if key not in ignored
        }
        for row in payload["filters"]["rows"]
    ]


def af2_lines(log_path: Path) -> dict:
    lines = log_path.read_text(errors="replace").splitlines()
    records = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\d+) models .* loss ([0-9.-]+)", line)
        if not match:
            continue
        sequence = next(
            (
                item.removeprefix("Sequence: ").strip()
                for item in lines[index + 1 : index + 4]
                if item.startswith("Sequence: ")
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
    return {
        "count": len(records),
        "records": records,
    }


def final_af2_metrics(payload: dict) -> dict | None:
    rows = payload["filters"]["rows"]
    if not rows:
        return None
    return {
        "plddt": float(rows[0]["plddt"]),
        "i_ptm": float(rows[0]["i_ptm"]),
        "i_pae": float(rows[0]["i_pae"]),
    }


def load_boundary(run_name: str) -> dict:
    manifest_path = (
        RESULTS / run_name / "boundary_snapshots/"
        "boundary_snapshot_manifest.json"
    )
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file()
        else {}
    )
    entries = {
        item["key"]: item for item in manifest.get("captured", [])
    }
    return {
        "manifest": {
            "path": str(manifest_path.relative_to(REPO_ROOT)),
            "sha256": sha256(manifest_path),
        }
        if manifest_path.is_file()
        else None,
        "run_finished": manifest.get("run_finished"),
        "af2_pdb_sha256": entries.get("af2_trajectory_pdb", {}).get(
            "preserved_sha256"
        ),
        "af3_input_sha256": entries.get("af3_input_json", {}).get(
            "preserved_sha256"
        ),
        "af3_msa_sha256": entries.get("af3_target_msa", {}).get(
            "preserved_sha256"
        ),
        "initial_af3_pdb_sha256": entries.get("initial_af3_pdb", {}).get(
            "preserved_sha256"
        ),
        "initial_relaxed_pdb_sha256": entries.get(
            "initial_relaxed_pdb", {}
        ).get("preserved_sha256"),
        "abmpnn_input_sha256": entries.get("abmpnn_input_pdb", {}).get(
            "preserved_sha256"
        ),
        "abmpnn_settings_sha256": entries.get(
            "abmpnn_run_settings", {}
        ).get("preserved_sha256"),
        "entries": entries,
    }


def worker_identities(run_name: str) -> list[dict]:
    structures = (
        RESULTS
        / run_name
        / "results/germinal_run/trajectories/structures"
    )
    return [
        json.loads(path.read_text())
        for path in sorted(
            structures.glob("abmpnn_workers/*/trial_*.runtime.json")
        )
    ]


def abmpnn_settings(run_name: str) -> dict | None:
    design = "il3_nb_s1791095845"
    path = (
        RESULTS
        / run_name
        / "results/germinal_run/trajectories/structures/"
        f"abmpnn_workers/{design}/run_settings.json"
    )
    if not path.is_file():
        return None
    settings = json.loads(path.read_text())
    settings.pop("project_dir", None)
    return settings


def compare_runs(left: dict, right: dict) -> dict:
    left_inputs = {
        item["name"]: item["input_json"]["sha256"]
        for item in left["af3"]["invocations"]
    }
    right_inputs = {
        item["name"]: item["input_json"]["sha256"]
        for item in right["af3"]["invocations"]
    }
    left_pdbs = {
        item["path"]: item["sha256"]
        for item in left["pdb_validation"]["files"]
    }
    right_pdbs = {
        item["path"]: item["sha256"]
        for item in right["pdb_validation"]["files"]
    }
    left_boundary = load_boundary("run1")
    right_boundary = load_boundary("run2")
    left_identities = worker_identities("run1")
    right_identities = worker_identities("run2")
    left_search_data = {
        item["name"]: item["consumed_search_data"]
        for item in left["af3"]["invocations"]
    }
    right_search_data = {
        item["name"]: item["consumed_search_data"]
        for item in right["af3"]["invocations"]
    }
    checks = {
        "run1_exit_zero": left["exit_code"] == 0,
        "run2_exit_zero": right["exit_code"] == 0,
        "normalized_configuration_equal": (
            normalized_config(left["config"])
            == normalized_config(right["config"])
        ),
        "af2_105_records_equal": (
            left["af2"]["iteration_records"]
            == right["af2"]["iteration_records"]
            and left["af2"]["iteration_record_count"] == 105
            and right["af2"]["iteration_record_count"] == 105
        ),
        "af2_final_sequence_equal": (
            left["af2"]["final_sequence"]
            == right["af2"]["final_sequence"]
            == AF2_SEQUENCE
        ),
        "af2_boundary_pdb_equal": (
            left_boundary["af2_pdb_sha256"]
            == right_boundary["af2_pdb_sha256"]
        ),
        "abmpnn_input_pdb_equal": (
            left_boundary["abmpnn_input_sha256"]
            == right_boundary["abmpnn_input_sha256"]
        ),
        "abmpnn_settings_equal_after_path_normalization": (
            abmpnn_settings("run1")
            == abmpnn_settings("run2")
        ),
        "abmpnn_records_equal": (
            left["abmpnn"]["records"] == right["abmpnn"]["records"]
        ),
        "abmpnn_sequence_set_expected": (
            len(left["filters"]["rows"]) == 5
            and len(right["filters"]["rows"]) == 5
            and hashlib.sha256(
                json.dumps(
                    [row["trajectory_sequence"] for row in left["filters"]["rows"]],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            == EXPECTED_SEQUENCE_SHA256
            and hashlib.sha256(
                json.dumps(
                    [row["trajectory_sequence"] for row in right["filters"]["rows"]],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            == EXPECTED_SEQUENCE_SHA256
        ),
        "af3_inputs_equal": left_inputs == right_inputs,
        "af3_database_search_data_equal": (
            left_search_data == right_search_data
        ),
        "af3_target_msa_equal": (
            left_boundary["af3_msa_sha256"]
            == right_boundary["af3_msa_sha256"]
        ),
        "initial_af3_structure_equal": (
            left_boundary["initial_af3_pdb_sha256"]
            == right_boundary["initial_af3_pdb_sha256"]
        ),
        "initial_relaxed_structure_equal": (
            left_boundary["initial_relaxed_pdb_sha256"]
            == right_boundary["initial_relaxed_pdb_sha256"]
        ),
        "af3_invocation_count": (
            left["af3"]["invocation_count"]
            == right["af3"]["invocation_count"]
            == 6
        ),
        "filter_rows_equal": (
            normalized_filter_rows(left) == normalized_filter_rows(right)
        ),
        "filter_decisions_equal": (
            left["filters"]["accepted_count"]
            == right["filters"]["accepted_count"]
            and left["filters"]["rejected_count"]
            == right["filters"]["rejected_count"]
        ),
        "pdb_inventory_equal": (
            left_pdbs == right_pdbs and len(left_pdbs) == 18
        ),
        "all_pdbs_parse": (
            left["pdb_validation"]["all_parse_success"]
            and right["pdb_validation"]["all_parse_success"]
        ),
        "all_coordinates_finite": (
            left["pdb_validation"]["all_finite"]
            and right["pdb_validation"]["all_finite"]
        ),
        "expected_chains": (
            left["pdb_validation"]["all_expected_chains"]
            and right["pdb_validation"]["all_expected_chains"]
        ),
        "worker_runtimes_correct": (
            len(left_identities) > 0
            and len(right_identities) > 0
            and all(
                item["jax_mps_version"] == "0.10.11"
                and item["sys_prefix"].endswith(
                    ".envs/abmpnn-mps-0111"
                )
                and item["jax_default_backend"] == "mps"
                for item in left_identities + right_identities
            )
        ),
        "worker_pids_distinct_across_runs": not (
            {item["pid"] for item in left_identities}
            & {item["pid"] for item in right_identities}
        ),
    }
    return {
        "phase": "2C",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run1": {
            "exit_code": left["exit_code"],
            "runtime": left["runtime"],
            "wall_seconds": left["wall_seconds"],
            "af2_iterations": left["af2"]["iteration_record_count"],
            "abmpnn_sequences": len(left["filters"]["rows"]),
            "af3_invocations": left["af3"]["invocation_count"],
            "accepted": left["filters"]["accepted_count"],
            "rejected": left["filters"]["rejected_count"],
            "pdb_count": left["pdb_validation"]["count"],
            "boundary": left_boundary,
            "worker_identities": left_identities,
        },
        "run2": {
            "exit_code": right["exit_code"],
            "runtime": right["runtime"],
            "wall_seconds": right["wall_seconds"],
            "af2_iterations": right["af2"]["iteration_record_count"],
            "abmpnn_sequences": len(right["filters"]["rows"]),
            "af3_invocations": right["af3"]["invocation_count"],
            "accepted": right["filters"]["accepted_count"],
            "rejected": right["filters"]["rejected_count"],
            "pdb_count": right["pdb_validation"]["count"],
            "boundary": right_boundary,
            "worker_identities": right_identities,
        },
        "comparison_policy": json.loads(
            (DOCS / "phase2c_preflight.json").read_text()
        )["comparison_policy"],
    }


def main() -> int:
    reports = load_reporter()
    reports.RESULTS = RESULTS
    reports.RUNS = {name: RESULTS / name for name in RUN_NAMES}
    payloads = {}
    for name in RUN_NAMES:
        log_path = RESULTS / f"{name}.log"
        if not log_path.is_file():
            raise SystemExit(f"Missing Phase 2C log: {log_path}")
        payload = reports.parse_run(name, RESULTS / name)
        payload["af2_detail"] = af2_lines(log_path)
        payloads[name] = payload
        write_json(DOCS / f"phase2c_{name}.json", payload)
    comparison = compare_runs(
        payloads["run1"], payloads["run2"]
    )
    run1_complete = payloads["run1"]["exit_code"] == 0
    run2_complete = payloads["run2"]["exit_code"] == 0
    if not run1_complete and not run2_complete:
        comparison["status"] = "FAIL"
    elif run1_complete != run2_complete:
        comparison["status"] = "PARTIAL"
    elif all(comparison["checks"].values()):
        comparison["status"] = "PASS"
    else:
        comparison["status"] = "FAIL"
    comparison["historical_af2"] = {
        "run1_iterations": payloads["run1"]["af2"][
            "iteration_record_count"
        ],
        "run2_iterations": payloads["run2"]["af2"][
            "iteration_record_count"
        ],
        "run1_final_sequence_exact": (
            payloads["run1"]["af2"]["final_sequence"] == AF2_SEQUENCE
        ),
        "run2_final_sequence_exact": (
            payloads["run2"]["af2"]["final_sequence"] == AF2_SEQUENCE
        ),
        "run1_final_metrics": final_af2_metrics(payloads["run1"]),
        "run2_final_metrics": final_af2_metrics(payloads["run2"]),
        "historical_metrics": {
            "plddt": 0.8867384791374207,
            "i_ptm": 0.742160439491272,
            "i_pae": 0.21023310720920563,
        },
    }
    comparison["historical_af2"]["run1_metrics_exact"] = (
        comparison["historical_af2"]["run1_final_metrics"]
        == comparison["historical_af2"]["historical_metrics"]
    )
    comparison["historical_af2"]["run2_metrics_exact"] = (
        comparison["historical_af2"]["run2_final_metrics"]
        == comparison["historical_af2"]["historical_metrics"]
    )
    if (
        comparison["status"] == "PASS"
        and not (
            comparison["historical_af2"]["run1_metrics_exact"]
            and comparison["historical_af2"]["run2_metrics_exact"]
        )
    ):
        comparison["status"] = "FAIL"
    write_json(
        DOCS / "phase2c_repeatability_comparison.json",
        comparison,
    )
    print(
        json.dumps(
            {"status": comparison["status"], "checks": comparison["checks"]},
            indent=2,
        )
    )
    return 0 if comparison["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
