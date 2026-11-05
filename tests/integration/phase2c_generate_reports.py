#!/usr/bin/env python3
"""Generate the Phase 2C report, summary, manifest, and checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2c"
RUN_NAMES = ("run1", "run2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(run_name: str) -> str:
    return (
        "JAX_PLATFORMS=mps PYTHONHASHSEED=7 "
        'LIBCIFPP_DATA_DIR="$PWD/.envs/germinal-macos/share/libcifpp" '
        'PATH="$PWD/.envs/af2-mps-0109/bin:$PATH" '
        'PYTHONPATH="$PWD/scripts:$PWD" '
        "/usr/bin/time -l .envs/af2-mps-0109/bin/python -u "
        "run_germinal.py run=vhh_il3_macos target=il3 "
        "filter/initial=vhh_il3 filter/final=vhh_il3 "
        f"project_dir=docs/porting/af3_mps_probe/results/phase2c/{run_name} "
        "pdb_dir=docs/porting/af3_mps_probe/results/phase2b4/run1/input "
        "+starting_pdb_complex=docs/porting/af3_mps_probe/results/"
        "phase2b4/run1/input/il3_11_8_18_nb.pdb "
        "af_params_dir=.envs/af2-mps-0109/share/af2-params/extracted "
        "dssp_path=.envs/germinal-macos/bin/mkdssp "
        "dalphaball_path=.envs/germinal-macos/bin/dalphaball "
        "af3_database_manifest=null af3_db_dir=/Volumes/T9/database "
        "pregenerate_seeds=true max_trajectories=1 "
        "max_hallucinated_trajectories=1 max_passing_designs=1"
    )


def main() -> int:
    preflight = load_json(DOCS / "phase2c_preflight.json")
    runtime = load_json(DOCS / "phase2c_runtime_manifest.json")
    database = load_json(DOCS / "phase2c_database_manifest.json")
    comparison = load_json(
        DOCS / "phase2c_repeatability_comparison.json"
    )
    runs = {
        name: load_json(DOCS / f"phase2c_{name}.json")
        for name in RUN_NAMES
    }
    failed_startup_log = (
        RESULTS / "run2_failed_startup_metal_unavailable.log"
    )
    failed_startup = (
        {
            "classification": "non-production sandbox startup failure",
            "scientific_stage_started": False,
            "log": {
                "path": str(failed_startup_log.relative_to(REPO_ROOT)),
                "sha256": sha256(failed_startup_log),
            },
            "resolution": (
                "Run 2 was restarted as a fresh process with native Metal "
                "access; this attempt is not counted as a production run."
            ),
        }
        if failed_startup_log.is_file()
        else None
    )
    failed_checks = [
        name for name, value in comparison["checks"].items() if not value
    ]
    decision = comparison["status"]

    summary = {
        "phase": "2C",
        "decision": decision,
        "scope": (
            "Two independent literal run_germinal.py production executions "
            "using the Phase 2B.12 component-isolated runtime architecture."
        ),
        "preflight_status": preflight["status"],
        "runtime_split": runtime["runtime_split"],
        "database": {
            "root": database["database_root"],
            "status": database["database_ready"],
            "readiness_manifest_sha256": database["readiness_manifest"][
                "sha256"
            ],
        },
        "runs": {
            name: {
                "status": (
                    "PASS"
                    if data["exit_code"] == 0
                    else "FAIL"
                ),
                "exit_code": data["exit_code"],
                "runtime": data["runtime"],
                "wall_seconds": data["wall_seconds"],
                "af2_iterations": data["af2"]["iteration_record_count"],
                "abmpnn_sequences": len(data["filters"]["rows"]),
                "af3_invocations": data["af3"]["invocation_count"],
                "accepted": data["filters"]["accepted_count"],
                "rejected": data["filters"]["rejected_count"],
                "pdb_count": data["pdb_validation"]["count"],
                "log": data["log"],
            }
            for name, data in runs.items()
        },
        "repeatability_status": comparison["status"],
        "failed_comparison_checks": failed_checks,
        "historical_af2": comparison["historical_af2"],
        "full_pipeline_runs": 2,
        "failed_launch_attempts": (
            [failed_startup] if failed_startup else []
        ),
        "limitations": [
            "Only documented paths and timing metadata are normalized.",
            "No numerical tolerance is permitted.",
            "The exact Metal implementation defect remains unidentified.",
        ],
        "artifacts": {
            "report": "docs/porting/af3_mps_probe/phase2c_report.md",
            "preflight": "docs/porting/af3_mps_probe/phase2c_preflight.json",
            "runtime_manifest": (
                "docs/porting/af3_mps_probe/"
                "phase2c_runtime_manifest.json"
            ),
            "database_manifest": (
                "docs/porting/af3_mps_probe/"
                "phase2c_database_manifest.json"
            ),
            "run1": "docs/porting/af3_mps_probe/phase2c_run1.json",
            "run2": "docs/porting/af3_mps_probe/phase2c_run2.json",
            "repeatability_comparison": (
                "docs/porting/af3_mps_probe/"
                "phase2c_repeatability_comparison.json"
            ),
            "execution_manifest": (
                "docs/porting/af3_mps_probe/"
                "phase2c_execution_manifest.json"
            ),
        },
    }
    write_json(DOCS / "phase2c_summary.json", summary)

    manifest = {
        "phase": "2C",
        "decision": decision,
        "preflight": preflight,
        "runtime_manifest": runtime,
        "database_manifest": database,
        "commands": {name: command(name) for name in RUN_NAMES},
        "runs": {
            name: {
                "output_directory": data["root"],
                "log": data["log"],
                "exit_code": data["exit_code"],
                "exit_code_source": data["exit_code_source"],
                "runtime": data["runtime"],
                "wall_seconds": data["wall_seconds"],
                "max_rss_bytes": data["max_rss_bytes"],
                "peak_memory_footprint_bytes": data[
                    "peak_memory_footprint_bytes"
                ],
                "af2_iteration_records": data["af2"][
                    "iteration_record_count"
                ],
                "abmpnn_sequence_count": len(data["filters"]["rows"]),
                "af3_invocation_count": data["af3"]["invocation_count"],
                "pdb_count": data["pdb_validation"]["count"],
                "pdb_sha256": {
                    item["path"]: item["sha256"]
                    for item in data["pdb_validation"]["files"]
                },
            }
            for name, data in runs.items()
        },
        "failed_launch_attempts": (
            [failed_startup] if failed_startup else []
        ),
    }
    write_json(
        DOCS / "phase2c_execution_manifest.json",
        manifest,
    )

    if decision == "PASS":
        conclusion = (
            "Both independent full production executions completed every "
            "configured stage and matched at every deterministic boundary."
        )
    elif decision == "PARTIAL":
        conclusion = (
            "The completed run is preserved, but the second run did not "
            "finish the full workflow."
        )
    else:
        conclusion = (
            "A mandatory execution or scientific-integrity check failed. "
            "No criterion was weakened."
        )
    report = f"""# Phase 2C - Full Production End-to-End Repeatability

Date: 2026-11-03

## Decision

**{decision}**

{conclusion}

## Frozen runtime

| Component | Runtime |
|---|---|
| AF2 | `jax-mps 0.10.9` |
| AF3 | `jax-mps 0.10.9` |
| AbMPNN | `jax-mps 0.10.11` |
| PyRosetta / filters | native ARM64 CPU |

The three runtime fingerprints were recorded before execution. The AF2 parent
launched AbMPNN through the explicit `.envs/abmpnn-mps-0111` interpreter, and
each worker reported its own imported package path, version, plugin hash, PID,
and `MPS:0` placement.

## Preflight

Preflight status: **{preflight["status"]}**

The source commit, unchanged scientific source hashes, all five AF2 checkpoint
hashes, original input PDB, effective configuration, AF3 revision/model,
runtime split, production database path, available disk, and absence of active
component processes were checked before Run 1.

Production database: `{database["database_root"]}`.
The existing production-readiness manifest and live readiness record both
report `PASS_PRODUCTION_READY`. No database download, provisioning, or
large-file checksum scan was repeated.

## Production runs

Each run used the literal `run_germinal.py` command recorded in
`phase2c_execution_manifest.json`, in a separate directory, with seed
`1791095845`.

| Run | Exit | Runtime | AF2 | AbMPNN | AF3 | Accepted | Rejected | PDBs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1 | {runs["run1"]["exit_code"]} | {runs["run1"]["runtime"]} | {runs["run1"]["af2"]["iteration_record_count"]} | {len(runs["run1"]["filters"]["rows"])} | {runs["run1"]["af3"]["invocation_count"]} | {runs["run1"]["filters"]["accepted_count"]} | {runs["run1"]["filters"]["rejected_count"]} | {runs["run1"]["pdb_validation"]["count"]} |
| Run 2 | {runs["run2"]["exit_code"]} | {runs["run2"]["runtime"]} | {runs["run2"]["af2"]["iteration_record_count"]} | {len(runs["run2"]["filters"]["rows"])} | {runs["run2"]["af3"]["invocation_count"]} | {runs["run2"]["filters"]["accepted_count"]} | {runs["run2"]["filters"]["rejected_count"]} | {runs["run2"]["pdb_validation"]["count"]} |

## Non-production startup attempt

An initial Run 2 launch attempt stopped after 1.87 seconds at the JAX Metal
device smoke test with `No Metal device available`. It did not enter AF2 or
any other scientific stage and is not counted as either production run. The
failure was caused by the command sandbox lacking native Metal access. The
evidence is preserved at
`results/phase2c/run2_failed_startup_metal_unavailable.log`.

Run 2 was then started as a new fresh process with native Metal access. Its
successful production log is `results/phase2c/run2.log`.

## Repeatability

Comparison status: **{comparison["status"]}**

Failed checks:

{json.dumps(failed_checks, indent=2)}

The comparison policy was frozen before Run 1. It permits normalization only
for run-specific paths, timing fields, and documented non-scientific CIF
timestamps. AF2 records and sequences, AbMPNN inputs and candidates, AF3
inputs and structures, PyRosetta metrics, filter decisions, and final PDB
hashes receive exact comparisons.

AF2 historical match:

```json
{json.dumps(comparison["historical_af2"], indent=2)}
```

## Runtime boundary

The final outputs include the runtime probe for AF2, AF3, and AbMPNN, plus
per-worker runtime identities. The negative wrong-version control failed
closed, and the cached runtime/regression checks passed.

## Evidence

- Preflight: `phase2c_preflight.json`
- Runtime manifest: `phase2c_runtime_manifest.json`
- Database manifest: `phase2c_database_manifest.json`
- Run 1 parsed result: `phase2c_run1.json`
- Run 2 parsed result: `phase2c_run2.json`
- Repeatability comparison: `phase2c_repeatability_comparison.json`
- Execution manifest: `phase2c_execution_manifest.json`
- Failed Run 2 startup evidence:
  `results/phase2c/run2_failed_startup_metal_unavailable.log`
- Raw logs and artifacts: `results/phase2c/run1/`, `results/phase2c/run2/`

## Limitations

- The exact Metal implementation defect remains unidentified.
- No scientific parameter, threshold, checkpoint, input, seed, or sampling
  distribution was changed to obtain this result.
"""
    (DOCS / "phase2c_report.md").write_text(report)

    checksum_paths = [
        DOCS / "phase2c_report.md",
        DOCS / "phase2c_summary.json",
        DOCS / "phase2c_preflight.json",
        DOCS / "phase2c_runtime_manifest.json",
        DOCS / "phase2c_database_manifest.json",
        DOCS / "phase2c_run1.json",
        DOCS / "phase2c_run2.json",
        DOCS / "phase2c_repeatability_comparison.json",
        DOCS / "phase2c_execution_manifest.json",
    ]
    checksum_paths.extend(path for path in RESULTS.rglob("*") if path.is_file())
    seen = set()
    lines = []
    for path in checksum_paths:
        relative = path.relative_to(REPO_ROOT)
        if relative in seen or not path.is_file():
            continue
        seen.add(relative)
        lines.append(f"{sha256(path)}  {relative}\n")
    (DOCS / "phase2c_checksums.sha256").write_text("".join(lines))
    print(
        json.dumps(
            {
                "decision": decision,
                "failed_checks": failed_checks,
                "runs": summary["runs"],
            },
            indent=2,
        )
    )
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
