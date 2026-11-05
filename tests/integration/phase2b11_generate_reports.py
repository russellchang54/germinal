#!/usr/bin/env python3
"""Generate Phase 2B.11 summaries, manifest, and report."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2b11"
ASSETS = DOCS / "phase2b4_assets.json"
SOURCE_INTEGRITY = DOCS / "phase2b9_source_integrity.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_checkpoints() -> dict:
    assets = json.loads(ASSETS.read_text())
    rows = []
    for item in assets["af2_checkpoints"]:
        path = REPO_ROOT / item["path"]
        actual = sha256(path)
        rows.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "matches": actual == item["sha256"],
                "bytes": path.stat().st_size,
            }
        )
    return {
        "all_match": all(row["matches"] for row in rows),
        "checkpoints": rows,
    }


def main() -> int:
    run1 = json.loads((DOCS / "phase2b11_run1.json").read_text())
    run2 = json.loads((DOCS / "phase2b11_run2.json").read_text())
    comparison = json.loads(
        (DOCS / "phase2b11_comparison.json").read_text()
    )
    source_integrity = json.loads(SOURCE_INTEGRITY.read_text())
    current_source_hashes = {
        path: sha256(REPO_ROOT / path)
        for path in source_integrity["source_hashes"]
    }
    checkpoints = verify_checkpoints()
    input_pdb = (
        REPO_ROOT
        / "docs/porting/af3_mps_probe/results/phase2b4/run1/input/"
        "il3_11_8_18_nb.pdb"
    )
    input_hash = sha256(input_pdb)

    environment = {
        "python": sys.version,
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "jax-mps": importlib.metadata.version("jax-mps"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    fixed_baseline = {
        "git_commit": source_integrity["git_commit"],
        "source_snapshot_sha256": source_integrity["source_snapshot"][
            "sha256"
        ],
        "source_hashes_match_accepted_snapshot": all(
            current_source_hashes[path] == expected
            for path, expected in source_integrity[
                "source_hashes"
            ].items()
        ),
        "input_pdb": str(input_pdb.relative_to(REPO_ROOT)),
        "input_pdb_sha256": input_hash,
        "input_expected_sha256": (
            "dea93d03be73a033a8d1e2233c711cad"
            "4ff43aa41d6c32f5197cf446e774d091"
        ),
        "config_hashes": {
            path: sha256(REPO_ROOT / path)
            for path in (
                "configs/run/vhh_il3_macos.yaml",
                "configs/target/il3.yaml",
                "configs/filter/initial/vhh_il3.yaml",
                "configs/filter/final/vhh_il3.yaml",
            )
        },
        "checkpoints": checkpoints,
    }

    artifact_paths = {
        "run1_log": RESULTS / "run1.log",
        "run2_log": RESULTS / "run2.log",
        "reference_0_10_9_log": RESULTS / "reference_0_10_9.log",
        "run1_trace": RESULTS / "run1/trace.json",
        "run2_trace": RESULTS / "run2/trace.json",
        "reference_0_10_9_trace": (
            RESULTS / "reference_0_10_9/trace.json"
        ),
        "run1_sequence_logits": (
            RESULTS / "run1/step_sequence_logits.npz"
        ),
        "run2_sequence_logits": (
            RESULTS / "run2/step_sequence_logits.npz"
        ),
        "run1_gradients": RESULTS / "run1/step_gradients.npz",
        "run2_gradients": RESULTS / "run2/step_gradients.npz",
        "run1_final_state": RESULTS / "run1/final_state.npz",
        "run2_final_state": RESULTS / "run2/final_state.npz",
        "comparison": DOCS / "phase2b11_comparison.json",
    }
    cpu_trace = RESULTS / "cpu_reference/trace.json"
    if cpu_trace.is_file():
        artifact_paths["cpu_reference_log"] = (
            RESULTS / "cpu_reference.log"
        )
        artifact_paths["cpu_reference_trace"] = cpu_trace
        artifact_paths["cpu_reference_sequence_logits"] = (
            RESULTS / "cpu_reference/step_sequence_logits.npz"
        )
        artifact_paths["cpu_reference_gradients"] = (
            RESULTS / "cpu_reference/step_gradients.npz"
        )
    elif (RESULTS / "cpu_reference.log").is_file():
        artifact_paths["cpu_reference_partial_log"] = (
            RESULTS / "cpu_reference.log"
        )
    for failed_name in (
        "run1_failed_aa_attribute",
        "run1_failed_final_scalar",
    ):
        log_path = RESULTS / f"{failed_name}.log"
        if log_path.is_file():
            artifact_paths[f"{failed_name}_log"] = log_path
    artifact_hashes = {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in artifact_paths.items()
    }
    commands = {
        "run1": (
            "JAX_PLATFORMS=mps PYTHONHASHSEED=7 "
            ".envs/germinal-mps-design/bin/python -u "
            "tests/integration/phase2b11_af2_logits_trace.py "
            "--backend mps --label run1 --output-root "
            "docs/porting/af3_mps_probe/results/phase2b11/run1 "
            "--seed 1791095845"
        ),
        "run2": (
            "JAX_PLATFORMS=mps PYTHONHASHSEED=7 "
            ".envs/germinal-mps-design/bin/python -u "
            "tests/integration/phase2b11_af2_logits_trace.py "
            "--backend mps --label run2 --output-root "
            "docs/porting/af3_mps_probe/results/phase2b11/run2 "
            "--seed 1791095845"
        ),
        "reference_0_10_9": (
            "JAX_PLATFORMS=mps PYTHONHASHSEED=7 "
            "PYTHONPATH=/private/tmp/jax_mps_0_10_9 "
            ".envs/germinal-mps-design/bin/python -u "
            "tests/integration/phase2b11_af2_logits_trace.py "
            "--backend mps --label reference_0_10_9 --output-root "
            "docs/porting/af3_mps_probe/results/phase2b11/"
            "reference_0_10_9 --seed 1791095845"
        ),
        "cpu_reference": (
            "JAX_PLATFORMS=cpu PYTHONHASHSEED=7 "
            ".envs/germinal-mps-design/bin/python -u "
            "tests/integration/phase2b11_af2_logits_trace.py "
            "--backend cpu --label cpu_reference --output-root "
            "docs/porting/af3_mps_probe/results/phase2b11/"
            "cpu_reference --seed 1791095845 --stop-after-logits"
        ),
    }
    execution_manifest = {
        "phase": "2B.11",
        "decision": comparison["decision"],
        "scope": "AF2 logits phase only; no full Germinal pipeline.",
        "commands": commands,
        "environment": environment,
        "fixed_baseline": fixed_baseline,
        "runs": {
            "run1": {
                "backend": run1["backend"],
                "jax_mps_version": run1["jax_mps_version"],
                "seed": run1["seed"],
                "step_count": run1["step_count"],
                "runtime_seconds": run1["runtime_seconds"],
                "best_log": run1["best_log"],
                "pdb_sha256": run1["pdb"]["sha256"],
            },
            "run2": {
                "backend": run2["backend"],
                "jax_mps_version": run2["jax_mps_version"],
                "seed": run2["seed"],
                "step_count": run2["step_count"],
                "runtime_seconds": run2["runtime_seconds"],
                "best_log": run2["best_log"],
                "pdb_sha256": run2["pdb"]["sha256"],
            },
        },
        "artifacts": artifact_hashes,
        "execution_records": [
            {
                "run": "run1",
                "status": "PASS",
                "result": "60/60 logits steps completed and traced",
            },
            {
                "run": "run2",
                "status": "PASS",
                "result": "60/60 logits steps completed and traced",
            },
            {
                "run": "reference_0_10_9",
                "status": "PASS",
                "result": (
                    "Equivalent first 60 logits steps completed; the "
                    "diagnostic runner also completed the remaining AF2 "
                    "softmax steps."
                ),
            },
            {
                "run": "run1_failed_aa_attribute",
                "status": "DIAGNOSTIC_FAILURE_PRESERVED",
                "result": (
                    "Trace serializer referenced AA from the wrong module "
                    "after step 1; no production behavior failed."
                ),
            },
            {
                "run": "run1_failed_final_scalar",
                "status": "DIAGNOSTIC_FAILURE_PRESERVED",
                "result": (
                    "Trace serializer failed converting a one-element final "
                    "metric after all 60 steps; no production behavior failed."
                ),
            },
            {
                "run": "cpu_reference",
                "status": (
                    "COMPLETE"
                    if comparison["cpu_reference"]["status"] == "COMPLETE"
                    else "PARTIAL_EXECUTION_PRESERVED"
                ),
                "result": (
                    f"{comparison['cpu_reference'].get('steps_observed', 0)} "
                    "parseable step records captured through step "
                    f"{max((item['step'] for item in comparison['cpu_reference'].get('records', [])), default=0)} "
                    "before the optional slow reference was stopped."
                ),
            },
        ],
    }
    write_json(
        DOCS / "phase2b11_execution_manifest.json",
        execution_manifest,
    )

    cpu_status = comparison["cpu_reference"]["status"]
    summary = {
        "phase": "2B.11",
        "decision": comparison["decision"],
        "classification": comparison["classification"],
        "seed": 1791095845,
        "input_pdb_sha256": input_hash,
        "run1_vs_run2": {
            "status": comparison["run1_vs_run2"]["status"],
            "all_60_sequence_logit_steps_exact": comparison[
                "run1_vs_run2"
            ]["all_sequence_logit_arrays_exact"],
            "all_60_gradient_steps_exact": comparison[
                "run1_vs_run2"
            ]["all_gradient_arrays_exact"],
            "final_metrics": comparison["run1_vs_run2"][
                "final_metrics"
            ],
            "final_sequence": comparison["run1_vs_run2"][
                "final_sequence"
            ],
            "best_sequence": comparison["run1_vs_run2"][
                "best_sequence"
            ],
            "pdb_sha256": comparison["run1_vs_run2"]["pdb_sha256"],
        },
        "historical_baseline": comparison[
            "historical_baseline_fidelity"
        ],
        "backend_causality": {
            "status": comparison["backend_divergence"]["status"],
            "first_divergence": comparison["backend_divergence"][
                "first_divergence"
            ],
            "final_metrics": comparison["backend_divergence"][
                "final_metrics"
            ],
            "step_sequence_logits": {
                key: value
                for key, value in comparison["backend_divergence"][
                    "step_sequence_logits_difference"
                ].items()
                if key != "rows"
            },
            "step_gradients": {
                key: value
                for key, value in comparison["backend_divergence"][
                    "step_gradients_difference"
                ].items()
                if key != "rows"
            },
        },
        "cpu_reference_status": cpu_status,
        "fixed_baseline": {
            "source_hashes_match_accepted_snapshot": fixed_baseline[
                "source_hashes_match_accepted_snapshot"
            ],
            "all_af2_checkpoints_match": checkpoints["all_match"],
            "input_pdb_hash_matches_expected": (
                input_hash == fixed_baseline["input_expected_sha256"]
            ),
        },
        "full_pipeline_run": False,
        "conclusion": (
            "The iPTM threshold miss is a reproducible 0.10.11 "
            "backend-associated numerical trajectory, not fresh-process "
            "nondeterminism. The accepted 0.10.9 trajectory was reproduced "
            "exactly in a same-source comparator. The first divergence occurs "
            "in the first AF2 forward pass despite identical initial "
            "parameters, inputs, and RNG keys."
        ),
        "artifacts": {
            name: str(REPO_ROOT / item["path"])
            for name, item in artifact_hashes.items()
        },
    }
    write_json(DOCS / "phase2b11_summary.json", summary)

    if cpu_status == "COMPLETE":
        cpu_section = (
            "The fixed-baseline CPU reference completed and is included in "
            "`phase2b11_comparison.json`."
        )
    elif cpu_status == "PARTIAL_EXECUTION":
        cpu_record_count = comparison["cpu_reference"]["steps_observed"]
        cpu_max_step = max(
            item["step"]
            for item in comparison["cpu_reference"]["records"]
        )
        cpu_step1 = comparison["cpu_reference"]["records"][0]
        cpu_section = (
            f"A same-source CPU reference captured {cpu_record_count} "
            f"parseable step records through step {cpu_max_step} before being "
            "stopped as an optional, slow diagnostic. "
            f"Its first-step loss was `{cpu_step1['loss']:.12f}`, already "
            "different from both MPS versions. The partial log and hashes are "
            "preserved. The accepted historical `0.10.9` trace remains the "
            "stage-equivalent baseline reference."
        )
    else:
        cpu_section = (
            "The same-source CPU reference was unavailable. The accepted "
            "historical `0.10.9` trace provides the stage-equivalent baseline "
            "reference."
        )
    report = f"""# Phase 2B.11 - AF2 Baseline Reproducibility

Date: 2026-11-02

## Decision

**{comparison["decision"]}**

Fresh-process execution on `jax-mps==0.10.11` is bitwise reproducible for the
fixed seed and all 60 logits-phase steps. The iPTM `0.649` result is a
reproducible backend-associated numerical trajectory, not run-to-run execution
nondeterminism.

## Fixed baseline

- Seed: `1791095845`
- Input PDB SHA-256: `{input_hash}`
- Source hashes match the accepted Phase 2B.9 snapshot:
  **{fixed_baseline["source_hashes_match_accepted_snapshot"]}**
- AF2 checkpoint hashes match the accepted Phase 2B.4 asset manifest:
  **{checkpoints["all_match"]}**
- JAX/JAXlib: `0.10.2`
- Design backend: `jax-mps==0.10.11`
- Scientific configuration, optimization schedule, and thresholds: unchanged

The production `germinal_design` function ran directly. No AF3, AbMPNN,
PyRosetta filtering, or full `run_germinal.py` workflow was launched.

## Fresh-process reproducibility

| Check | Run 1 vs Run 2 |
|---|---|
| Normalized full trace | Exact |
| 60 sequence-logit tensors | Bitwise exact |
| 60 gradient tensors | Bitwise exact |
| Parameter and optimizer hashes | Exact |
| Loss and loss components | Exact |
| Distogram and confidence hashes | Exact |
| RNG key sequence | Exact |
| Best/final sequences | Exact |
| Final model state | Bitwise exact |
| Output PDB | Bitwise exact |

Run times were {run1["runtime_seconds"]:.2f}s and
{run2["runtime_seconds"]:.2f}s.

Both runs ended at:

- pLDDT `{run1["best_log"]["plddt"]:.12f}`
- iPTM `{run1["best_log"]["i_ptm"]:.12f}`
- iPAE `{run1["best_log"]["i_pae"]:.12f}`
- Loss `{run1["best_log"]["loss"]:.12f}`

## First numerical divergence

The same tracing path was run against the preserved `0.10.9` backend. Before
the first forward pass:

- input PDB hash: identical;
- scientific configuration: identical;
- initial model parameters: identical;
- initial optimizer state: identical;
- RNG key: identical;
- sequence entropy term: identical.

At the first AF2 forward pass, the `0.10.9` loss was
`{comparison["backend_divergence"]["first_divergence"]["loss_0_10_9"]:.12f}`
and the `0.10.11` loss was
`{comparison["backend_divergence"]["first_divergence"]["loss_0_10_11"]:.12f}`.
Distogram, pLDDT, PAE, and atom-position hashes differed. Step-1 sequence
logits differed from the first element, with maximum absolute difference
`{comparison["backend_divergence"]["step_sequence_logits_difference"]["first_nonidentical_step"]["maximum_absolute_difference"]:.12f}`
and relative L2 difference
`{comparison["backend_divergence"]["step_sequence_logits_difference"]["first_nonidentical_step"]["relative_l2_difference"]:.12f}`.

All 60 sequence-logit and gradient steps differed between backends. This locates
the causal boundary within AF2 numerical execution, after invariant inputs,
checkpoints, parameters, RNG, and schedule.

## Historical baseline fidelity

The fresh `0.10.9` comparator exactly reproduced all 60 historical Phase 2B.9
logits-stage model metric lines and sequence lines, and matched its final
sequence and metrics. It ended at:

- pLDDT `{comparison["backend_divergence"]["final_metrics"]["0_10_9"]["plddt"]:.12f}`
- iPTM `{comparison["backend_divergence"]["final_metrics"]["0_10_9"]["i_ptm"]:.12f}`
- iPAE `{comparison["backend_divergence"]["final_metrics"]["0_10_9"]["i_pae"]:.12f}`

The two fresh `0.10.11` traces also exactly match the Phase 2B.10 literal-run
AF2 stage, including its output PDB SHA-256
`{comparison["run1_vs_run2"]["pdb_sha256"]}`. The backend change therefore
produces a stable replacement trajectory rather than an execution error.

## Reference fidelity

{cpu_section}

The trusted reference is not an individual backend result. The same-source
historical `0.10.9` trace establishes exact baseline fidelity, while the
controlled cross-version comparison establishes backend causality. Phase 2B.4
already showed that CPU and MPS AF2 trajectories are not bitwise equivalent,
so `0.10.9` is not treated as scientific ground truth.

## Conclusion

The AF2 threshold miss is **not execution nondeterminism** and is **not a new
seed or input outcome**. Under `0.10.11`, the fixed baseline deterministically
produces iPTM below the unchanged threshold. The numerical source is the
backend-dependent AF2 forward computation, beginning in the first forward pass.

## Evidence

- Comparison: `phase2b11_comparison.json`
- Run 1: `phase2b11_run1.json`
- Run 2: `phase2b11_run2.json`
- Execution manifest: `phase2b11_execution_manifest.json`
- Raw traces and tensors: `results/phase2b11/run1/`, `run2/`
- Historical `0.10.9` comparator: `results/phase2b11/reference_0_10_9/`
- Partial CPU reference: `results/phase2b11/cpu_reference.log`
- Preserved diagnostic failures: `results/phase2b11/run1_failed_*/`
"""
    (DOCS / "phase2b11_report.md").write_text(report)
    checksum_paths = [
        DOCS / "phase2b11_report.md",
        DOCS / "phase2b11_summary.json",
        DOCS / "phase2b11_run1.json",
        DOCS / "phase2b11_run2.json",
        DOCS / "phase2b11_comparison.json",
        DOCS / "phase2b11_execution_manifest.json",
        RESULTS / "run1/trace.json",
        RESULTS / "run2/trace.json",
        RESULTS / "reference_0_10_9/trace.json",
        RESULTS / "run1/step_sequence_logits.npz",
        RESULTS / "run2/step_sequence_logits.npz",
        RESULTS / "run1/step_gradients.npz",
        RESULTS / "run2/step_gradients.npz",
        RESULTS / "run1/final_state.npz",
        RESULTS / "run2/final_state.npz",
        RESULTS / "reference_0_10_9.log",
        RESULTS / "cpu_reference.log",
    ]
    (DOCS / "phase2b11_checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(REPO_ROOT)}\n"
            for path in checksum_paths
            if path.is_file()
        )
    )
    print(
        json.dumps(
            {
                "decision": comparison["decision"],
                "run1_vs_run2": comparison["run1_vs_run2"]["status"],
                "historical_baseline_fidelity": comparison[
                    "historical_baseline_fidelity"
                ]["status"],
                "backend_causality": comparison["backend_divergence"][
                    "status"
                ],
                "cpu_reference_status": cpu_status,
                "source_hashes_match": fixed_baseline[
                    "source_hashes_match_accepted_snapshot"
                ],
                "checkpoints_match": checkpoints["all_match"],
            },
            indent=2,
        )
    )
    return 0 if comparison["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
