#!/usr/bin/env python3
"""Compare the fixed-seed AF2 traces and equivalent historical records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2b11"
RUN1 = RESULTS / "run1"
RUN2 = RESULTS / "run2"
REFERENCE_0_10_9 = RESULTS / "reference_0_10_9"
CPU_REFERENCE = RESULTS / "cpu_reference"
PHASE2B9_JSON = DOCS / "phase2b9_run1.json"
PHASE2B9_LOG = DOCS / "results/phase2b9/run1.log"
PHASE2B10_RUN = DOCS / "phase2b10_production_run.json"
PHASE2B10_LOG = DOCS / "results/phase2b10/production_run1.log"
PHASE2B4_CPU_REFERENCE = DOCS / "phase2b4_cpu_mps_comparison.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_trace(root: Path) -> dict:
    return json.loads((root / "trace.json").read_text())


def normalized_trace(trace: dict) -> dict:
    value = json.loads(json.dumps(trace))
    for key in ("label", "config", "config_sha256", "runtime_seconds"):
        value.pop(key, None)
    value.get("pdb", {}).pop("path", None)
    for step in value.get("steps", []):
        step.pop("runtime_seconds", None)
    return value


def scientific_config(root: Path) -> dict:
    config = yaml.safe_load(
        (root / "results/germinal_run/final_config.yaml").read_text()
    )["run_settings"]
    return {
        key: value
        for key, value in config.items()
        if key not in {"project_dir"}
    }


def load_npz(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / name) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def array_difference_tables(left: dict, right: dict) -> dict:
    rows = []
    for key in sorted(set(left) & set(right)):
        a = left[key]
        b = right[key]
        difference = np.asarray(a, dtype=np.float64) - np.asarray(
            b, dtype=np.float64
        )
        denominator = np.linalg.norm(np.asarray(a, dtype=np.float64))
        rows.append(
            {
                "step": int(key.removeprefix("step_")),
                "shape_equal": a.shape == b.shape,
                "bitwise_equal": bool(np.array_equal(a, b)),
                "exact_equal_fraction": float(np.mean(a == b)),
                "maximum_absolute_difference": float(
                    np.max(np.abs(difference))
                ),
                "relative_l2_difference": float(
                    np.linalg.norm(difference) / denominator
                ),
            }
        )
    first = next(
        (row for row in rows if not row["bitwise_equal"]),
        None,
    )
    return {
        "all_keys_equal": set(left) == set(right),
        "key_count": len(set(left) & set(right)),
        "first_nonidentical_step": first,
        "nonidentical_step_count": sum(
            not row["bitwise_equal"] for row in rows
        ),
        "maximum_absolute_difference_overall": max(
            (row["maximum_absolute_difference"] for row in rows),
            default=0.0,
        ),
        "maximum_relative_l2_difference_overall": max(
            (row["relative_l2_difference"] for row in rows),
            default=0.0,
        ),
        "rows": rows,
    }


def first_logits_metrics(text: str, count: int) -> list[dict]:
    lines = text.splitlines()
    records = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\d+) models .* loss ([0-9.-]+)", line)
        if not match or len(records) >= count:
            continue
        sequence = None
        for following in lines[index + 1 : index + 4]:
            if following.startswith("Sequence: "):
                sequence = following.removeprefix("Sequence: ").strip()
                break
        records.append(
            {
                "iteration": int(match.group(1)),
                "loss": float(match.group(2)),
                "line": line,
                "sequence": sequence,
            }
        )
    return records


def historical_fidelity(trace: dict, text: str, historical: dict) -> dict:
    current_records = first_logits_metrics(text, 60)
    return {
        "equivalent_logits_iterations": 60,
        "iteration_numbers_exact": [
            item["iteration"] for item in current_records
        ]
        == list(range(1, 61)),
        "model_metric_lines_exact": [
            item["line"] for item in current_records
        ]
        == historical["model_metric_lines"],
        "sequence_lines_exact": [
            item["sequence"] for item in current_records
        ]
        == historical["sequences"],
        "final_sequence_exact": (
            trace["final_sequence"] == historical["final_sequence"]
            or trace["best_sequence"] == historical["final_sequence"]
        ),
        "final_metrics_match": all(
            math.isclose(
                trace["best_log"][key],
                historical["metrics"][key],
                rel_tol=0.0,
                abs_tol=1e-3,
            )
            for key in ("plddt", "i_ptm", "i_pae")
        ),
    }


def parse_historical_log() -> dict:
    records = first_logits_metrics(PHASE2B9_LOG.read_text(), 60)
    final_line = [
        line
        for line in PHASE2B9_LOG.read_text().splitlines()
        if line.startswith("Final pLDDT/iPTM/iPAE for design")
    ][-1]
    values = re.search(
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        final_line,
    )
    phase2b9 = json.loads(PHASE2B9_JSON.read_text())
    return {
        "iteration_count": len(records),
        "model_metric_lines": [item["line"] for item in records],
        "sequences": [item["sequence"] for item in records],
        "final_sequence": phase2b9["af2"]["final_sequence"],
        "metrics": {
            "plddt": float(values.group(1)),
            "i_ptm": float(values.group(2)),
            "i_pae": float(values.group(3)),
        },
    }


def parse_partial_cpu_log() -> dict:
    path = RESULTS / "cpu_reference.log"
    if not path.is_file():
        return {
            "status": "UNAVAILABLE",
            "path": str(path.relative_to(REPO_ROOT)),
        }
    text = path.read_text(errors="replace")
    records = []
    pattern = re.compile(
        r'\{"step":\s*(\d+).*?"loss":\s*([0-9.eE+-]+).*?'
        r'"gradient_sha256":\s*"([0-9a-f]+)".*?'
        r'"parameter_sha256_after":\s*"([0-9a-f]+)"\}'
    )
    for match in pattern.finditer(text):
        records.append(
            {
                "step": int(match.group(1)),
                "loss": float(match.group(2)),
                "gradient_sha256": match.group(3),
                "parameter_sha256_after": match.group(4),
            }
        )
    return {
        "status": "PARTIAL_EXECUTION",
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256(path),
        "steps_observed": len(records),
        "records": records,
        "reason_stopped": (
            "Optional CPU reference was stopped after the captured prefix "
            "because it was roughly two orders of magnitude slower and did "
            "not affect the required MPS reproducibility decision."
        ),
    }


def main() -> int:
    run1 = load_trace(RUN1)
    run2 = load_trace(RUN2)
    reference = load_trace(REFERENCE_0_10_9)
    run1_logits = load_npz(RUN1, "step_sequence_logits.npz")
    run2_logits = load_npz(RUN2, "step_sequence_logits.npz")
    run1_gradients = load_npz(RUN1, "step_gradients.npz")
    run2_gradients = load_npz(RUN2, "step_gradients.npz")
    ref_logits = load_npz(REFERENCE_0_10_9, "step_sequence_logits.npz")
    ref_gradients = load_npz(
        REFERENCE_0_10_9, "step_gradients.npz"
    )
    phase2b10 = json.loads(PHASE2B10_RUN.read_text())
    phase2b10_text = PHASE2B10_LOG.read_text()
    phase2b10_records = first_logits_metrics(phase2b10_text, 60)

    run12_checks = {
        "same_seed": run1["seed"] == run2["seed"] == 1791095845,
        "same_input_pdb_sha256": (
            run1["input_pdb_sha256"] == run2["input_pdb_sha256"]
        ),
        "same_scientific_config": (
            scientific_config(RUN1) == scientific_config(RUN2)
        ),
        "same_initial_parameter_sha256": (
            run1["steps"][0]["parameter_sha256_before"]
            == run2["steps"][0]["parameter_sha256_before"]
        ),
        "same_normalized_trace": (
            normalized_trace(run1) == normalized_trace(run2)
        ),
        "same_sequence_logits_npz_sha256": (
            sha256(RUN1 / "step_sequence_logits.npz")
            == sha256(RUN2 / "step_sequence_logits.npz")
        ),
        "same_gradients_npz_sha256": (
            sha256(RUN1 / "step_gradients.npz")
            == sha256(RUN2 / "step_gradients.npz")
        ),
        "same_final_state_npz_sha256": (
            sha256(RUN1 / "final_state.npz")
            == sha256(RUN2 / "final_state.npz")
        ),
        "same_pdb_sha256": (
            run1["pdb"]["sha256"] == run2["pdb"]["sha256"]
        ),
    }
    run12_npz_exact = {
        "sequence_logits": {
            key: bool(np.array_equal(run1_logits[key], run2_logits[key]))
            for key in sorted(run1_logits)
        },
        "gradients": {
            key: bool(
                np.array_equal(run1_gradients[key], run2_gradients[key])
            )
            for key in sorted(run1_gradients)
        },
    }

    scientific_equal = scientific_config(RUN1) == scientific_config(
        REFERENCE_0_10_9
    )
    first_pass_0_10_9 = reference["passes"][0]
    first_pass_0_10_11 = run1["passes"][0]
    first_step_0_10_9 = reference["steps"][0]
    first_step_0_10_11 = run1["steps"][0]
    reference_keys = [
        key
        for record in reference["passes"]
        if record["step"] <= 60
        for key in record["key_sha256"]
    ]
    candidate_keys = [
        key
        for record in run1["passes"]
        if record["step"] <= 60
        for key in record["key_sha256"]
    ]
    first_sequence_divergence = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(reference["steps"], run1["steps"]),
                start=1,
            )
            if left["sequence_after"] != right["sequence_after"]
        ),
        None,
    )
    backend_divergence = {
        "same_input_pdb_sha256": (
            run1["input_pdb_sha256"]
            == reference["input_pdb_sha256"]
        ),
        "same_scientific_config": scientific_equal,
        "same_initial_parameter_sha256": (
            first_step_0_10_9["parameter_sha256_before"]
            == first_step_0_10_11["parameter_sha256_before"]
        ),
        "same_initial_optimizer_sha256": (
            first_step_0_10_9["optimizer_state_sha256_before"]
            == first_step_0_10_11["optimizer_state_sha256_before"]
        ),
        "same_first_pass_rng_key": (
            first_pass_0_10_9["key_sha256"]
            == first_pass_0_10_11["key_sha256"]
        ),
        "same_first_pass_sequence_entropy": (
            first_pass_0_10_9["loss_components"]["seq_ent"]
            == first_pass_0_10_11["loss_components"]["seq_ent"]
        ),
        "all_logits_phase_rng_keys_equal": reference_keys
        == candidate_keys,
        "first_sequence_divergence_step": first_sequence_divergence,
        "first_divergence": {
            "step": 1,
            "forward_pass": 1,
            "stage": (
                "AF2 forward outputs after identical model parameters "
                "and RNG key"
            ),
            "loss_0_10_9": first_pass_0_10_9["loss"],
            "loss_0_10_11": first_pass_0_10_11["loss"],
            "loss_absolute_difference": abs(
                first_pass_0_10_9["loss"]
                - first_pass_0_10_11["loss"]
            ),
            "distogram_hashes_equal": (
                first_pass_0_10_9["distogram"]["sha256"]
                == first_pass_0_10_11["distogram"]["sha256"]
            ),
            "plddt_hashes_equal": (
                first_pass_0_10_9["plddt"]["sha256"]
                == first_pass_0_10_11["plddt"]["sha256"]
            ),
            "pae_hashes_equal": (
                first_pass_0_10_9["pae"]["sha256"]
                == first_pass_0_10_11["pae"]["sha256"]
            ),
            "atom_positions_hashes_equal": (
                first_pass_0_10_9["atom_positions"]["sha256"]
                == first_pass_0_10_11["atom_positions"]["sha256"]
            ),
        },
        "step_sequence_logits_difference": array_difference_tables(
            ref_logits, run1_logits
        ),
        "step_gradients_difference": array_difference_tables(
            ref_gradients, run1_gradients
        ),
        "final_metrics": {
            "0_10_9": {
                key: reference["best_log"][key]
                for key in ("plddt", "i_ptm", "i_pae", "loss")
            },
            "0_10_11": {
                key: run1["best_log"][key]
                for key in ("plddt", "i_ptm", "i_pae", "loss")
            },
            "delta": {
                key: run1["best_log"][key] - reference["best_log"][key]
                for key in ("plddt", "i_ptm", "i_pae", "loss")
            },
        },
        "final_sequences_equal": (
            reference["best_sequence"] == run1["best_sequence"]
        ),
    }

    historical = parse_historical_log()
    historical_0_10_9 = historical_fidelity(
        reference,
        (RESULTS / "reference_0_10_9.log").read_text(),
        historical,
    )
    phase2b10_historical = {
        "iteration_records_exact": [
            {
                "iteration": item["iteration"],
                "loss": item["loss"],
                "sequence": item["sequence"],
            }
            for item in phase2b10_records
        ]
        == [
            {
                "iteration": item["iteration"],
                "loss": item["loss"],
                "sequence": item["sequence"],
            }
            for item in first_logits_metrics(
                (RESULTS / "run1.log").read_text(), 60
            )
        ],
        "pdb_sha256_exact": (
            run1["pdb"]["sha256"] == phase2b10["pdb"]["sha256"]
        ),
        "final_metrics_exact_to_logged_precision": (
            {
                key: round(run1["best_log"][key], 3)
                for key in ("plddt", "i_ptm", "i_pae")
            }
            == {
                "plddt": 0.897,
                "i_ptm": 0.649,
                "i_pae": 0.254,
            }
        ),
        "sequence_lines_exact": (
            [item["sequence"] for item in phase2b10_records]
            == [
                item["sequence"]
                for item in first_logits_metrics(
                    (RESULTS / "run1.log").read_text(), 60
                )
            ]
        ),
    }

    cpu_reference = None
    if (CPU_REFERENCE / "trace.json").is_file():
        cpu = load_trace(CPU_REFERENCE)
        cpu_logits = load_npz(CPU_REFERENCE, "step_sequence_logits.npz")
        cpu_gradients = load_npz(
            CPU_REFERENCE, "step_gradients.npz"
        )
        cpu_reference = {
            "status": "COMPLETE",
            "backend": cpu["backend"],
            "step_count": cpu["step_count"],
            "best_log": cpu["best_log"],
            "best_sequence": cpu["best_sequence"],
            "vs_0_10_11_logits": array_difference_tables(
                cpu_logits, run1_logits
            ),
            "vs_0_10_11_gradients": array_difference_tables(
                cpu_gradients, run1_gradients
            ),
            "vs_0_10_9_logits": array_difference_tables(
                cpu_logits, ref_logits
            ),
            "vs_0_10_9_gradients": array_difference_tables(
                cpu_gradients, ref_gradients
            ),
        }
    else:
        cpu_reference = parse_partial_cpu_log()
        if cpu_reference.get("records"):
            cpu_reference["comparison_to_mps_0_10_11"] = [
                {
                    "step": record["step"],
                    "loss_equal": (
                        record["loss"]
                        == run1["steps"][record["step"] - 1]["loss"]
                    ),
                    "gradient_hash_equal": (
                        record["gradient_sha256"]
                        == run1["steps"][record["step"] - 1][
                            "gradient"
                        ]["sha256"]
                    ),
                    "parameter_hash_equal": (
                        record["parameter_sha256_after"]
                        == run1["steps"][record["step"] - 1][
                            "parameter_sha256_after"
                        ]
                    ),
                }
                for record in cpu_reference["records"]
            ]

    run12_pass = all(run12_checks.values()) and all(
        all(values.values()) for values in run12_npz_exact.values()
    )
    historical_pass = all(
        value
        for key, value in historical_0_10_9.items()
        if key != "equivalent_logits_iterations"
    )
    phase2b10_pass = all(
        value
        for key, value in phase2b10_historical.items()
        if isinstance(value, bool)
    )
    backend_causality = all(
        backend_divergence[key]
        for key in (
            "same_input_pdb_sha256",
            "same_scientific_config",
            "same_initial_parameter_sha256",
            "same_initial_optimizer_sha256",
            "same_first_pass_rng_key",
            "same_first_pass_sequence_entropy",
            "all_logits_phase_rng_keys_equal",
        )
    ) and not backend_divergence["first_divergence"][
        "distogram_hashes_equal"
    ]
    comparison = {
        "phase": "2B.11",
        "decision": (
            "PASS"
            if (
                run12_pass
                and historical_pass
                and phase2b10_pass
                and backend_causality
            )
            else "FAIL"
        ),
        "classification": (
            "Reproducible backend-associated AF2 numerical divergence; "
            "not execution nondeterminism."
        ),
        "run1_vs_run2": {
            "status": "PASS" if run12_pass else "FAIL",
            "checks": run12_checks,
            "all_sequence_logit_arrays_exact": all(
                run12_npz_exact["sequence_logits"].values()
            ),
            "all_gradient_arrays_exact": all(
                run12_npz_exact["gradients"].values()
            ),
            "array_checks": run12_npz_exact,
            "final_metrics": run1["best_log"],
            "final_sequence": run1["final_sequence"],
            "best_sequence": run1["best_sequence"],
            "pdb_sha256": run1["pdb"]["sha256"],
        },
        "historical_baseline_fidelity": {
            "accepted_0_10_9_endpoint": historical_0_10_9,
            "phase2b10_0_10_11_endpoint": phase2b10_historical,
            "status": (
                "PASS" if historical_pass and phase2b10_pass else "FAIL"
            ),
        },
        "backend_divergence": {
            "status": "ESTABLISHED" if backend_causality else "UNRESOLVED",
            **backend_divergence,
        },
        "cpu_reference": cpu_reference,
        "external_reference_context": {
            "source": str(PHASE2B4_CPU_REFERENCE.relative_to(REPO_ROOT)),
            "scope": (
                "Earlier same-source, different-fixture CPU/MPS control from "
                "Phase 2B.4. It is used only to establish that cross-backend "
                "AF2 trajectories are not expected to be bitwise equivalent."
            ),
            "summary": json.loads(PHASE2B4_CPU_REFERENCE.read_text()),
        },
        "threshold_outcome": {
            "0_10_9": {
                "metrics": reference["best_log"],
                "passes_plddt": reference["best_log"]["plddt"] >= 0.75,
                "passes_i_ptm": reference["best_log"]["i_ptm"] >= 0.68,
                "passes_i_pae": reference["best_log"]["i_pae"] < 0.3,
            },
            "0_10_11": {
                "metrics": run1["best_log"],
                "passes_plddt": run1["best_log"]["plddt"] >= 0.75,
                "passes_i_ptm": run1["best_log"]["i_ptm"] >= 0.68,
                "passes_i_pae": run1["best_log"]["i_pae"] < 0.3,
            },
        },
    }
    write_json(DOCS / "phase2b11_comparison.json", comparison)
    write_json(DOCS / "phase2b11_run1.json", run1)
    write_json(DOCS / "phase2b11_run2.json", run2)
    print(
        json.dumps(
            {
                "decision": comparison["decision"],
                "run1_vs_run2": comparison["run1_vs_run2"]["status"],
                "historical_baseline_fidelity": comparison[
                    "historical_baseline_fidelity"
                ]["status"],
                "backend_divergence": comparison["backend_divergence"][
                    "status"
                ],
                "cpu_reference": cpu_reference["status"],
            },
            indent=2,
        )
    )
    return 0 if comparison["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
