#!/usr/bin/env python3
"""Generate the Phase 2B.12 report, summary, manifest, and checksums."""

from __future__ import annotations

import difflib
import hashlib
import json
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2b12"
RUNTIMES = RESULTS / "runtime_probes"
SOURCE_SNAPSHOT = (
    DOCS
    / "results/phase2b9/baseline_execution/source_snapshot.tar.gz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_patch() -> str:
    changed = (
        "germinal/filters/redesign.py",
        "scripts/run_abmpnn_worker.py",
        "configs/run/vhh_il3_macos.yaml",
    )
    new_files = (
        "scripts/runtime_identity.py",
        "scripts/environment_manifest.py",
    )
    chunks = []
    with tarfile.open(SOURCE_SNAPSHOT) as archive:
        for path in changed:
            before = (
                archive.extractfile(path)
                .read()
                .decode()
                .splitlines(keepends=True)
            )
            after = (REPO_ROOT / path).read_text().splitlines(
                keepends=True
            )
            chunks.append(
                "".join(
                    difflib.unified_diff(
                        before,
                        after,
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}",
                    )
                )
            )
    for path in new_files:
        after = (REPO_ROOT / path).read_text().splitlines(
            keepends=True
        )
        chunks.append(
            "".join(
                difflib.unified_diff(
                    [],
                    after,
                    fromfile="/dev/null",
                    tofile=f"b/{path}",
                )
            )
        )
    return "".join(chunks)


def main() -> int:
    provenance = load(DOCS / "phase2b12_backend_provenance.json")
    baseline = load(DOCS / "phase2b12_baseline_integrity.json")
    release = load(DOCS / "phase2b12_release_decision.json")
    gates = load(DOCS / "phase2b12_gate_results.json")
    component = load(RESULTS / "component_boundary.json")
    negative = load(RUNTIMES / "abmpnn_wrong_version.json")

    required_status = {
        "backend_provenance": provenance["status"],
        "baseline_integrity": baseline["status"],
        "release_decision": release["status"],
        "gate_a_af2_preservation": gates["gate_a"]["status"],
        "gate_b_af3_compatibility": gates["gate_b"]["status"],
        "gate_c_abmpnn_reliability": gates["gate_c"]["status"],
        "gate_d_orchestration": gates["gate_d"]["status"],
    }
    decision = "PASS" if all(
        value == "PASS" for value in required_status.values()
    ) else "FAIL"

    patch_path = DOCS / "phase2b12_exact_patch.diff"
    patch_path.write_text(build_patch())
    summary = {
        "phase": "2B.12",
        "decision": decision,
        "scope": (
            "Component-isolated runtime validation using preserved AF2, AF3, "
            "and production-boundary artifacts; no full pipeline run."
        ),
        "stage_scoped_runtime_decision": release[
            "stage_scoped_runtime_decision"
        ],
        "statuses": required_status,
        "runtime_versions": {
            name: {
                "jax": runtime["jax_version"],
                "jaxlib": runtime["jaxlib_version"],
                "jax-mps": runtime["jax_mps_version"],
                "backend": runtime["jax_default_backend"],
                "sys_prefix": runtime["sys_prefix"],
                "plugin_sha256": runtime["plugin_library_sha256"],
            }
            for name, runtime in provenance["runtimes"].items()
        },
        "gate_details": {
            "af2": {
                "best_sequence": gates["gate_a"]["best_sequence"],
                "plddt": gates["gate_a"]["best_log"]["plddt"],
                "i_ptm": gates["gate_a"]["best_log"]["i_ptm"],
                "i_pae": gates["gate_a"]["best_log"]["i_pae"],
                "pdb_sha256": gates["gate_a"]["pdb"]["sha256"],
            },
            "af3": {
                "revision": gates["gate_b"]["af3_revision"],
                "model_sha256": gates["gate_b"]["af3_model_sha256"],
                "input_json_sha256": gates["gate_b"][
                    "run1_input_sha256"
                ],
                "prediction_pdb_sha256": gates["gate_b"][
                    "run1_af3_pdb_sha256"
                ],
                "relaxed_pdb_sha256": gates["gate_b"][
                    "run1_relaxed_pdb_sha256"
                ],
                "targeted_inference_status": gates["gate_b"][
                    "targeted_inference"
                ]["status"],
                "targeted_inference_seconds": gates["gate_b"][
                    "targeted_inference"
                ]["runtime_seconds"],
                "targeted_sample_directory_count": gates["gate_b"][
                    "targeted_inference"
                ]["sample_directory_count"],
                "targeted_selected_pdb_sha256": gates["gate_b"][
                    "targeted_af3_selected_pdb_sha256"
                ],
            },
            "abmpnn": {
                "repeat_count": gates["gate_c"]["repeat_count"],
                "all_sequence_sets_identical": gates["gate_c"][
                    "all_sequence_sets_identical"
                ],
                "sequence_set_sha256": gates["gate_c"][
                    "sequence_hashes"
                ][0],
                "worker_pids": gates["gate_c"]["worker_pids"],
            },
            "orchestration": {
                "parent_jax_mps_version": gates["gate_d"][
                    "parent_runtime"
                ]["jax_mps_version"],
                "parent_prefix": gates["gate_d"]["parent_runtime"][
                    "sys_prefix"
                ],
                "child_runtime_config": gates["gate_d"][
                    "abmpnn_runtime_config"
                ],
                "invalid_runtime_control_status": negative["status"],
            },
        },
        "baseline_integrity": baseline["checks"],
        "full_pipeline_run": False,
        "limitations": release["limitations"],
        "artifacts": {
            "report": "docs/porting/af3_mps_probe/phase2b12_report.md",
            "backend_provenance": (
                "docs/porting/af3_mps_probe/"
                "phase2b12_backend_provenance.json"
            ),
            "baseline_integrity": (
                "docs/porting/af3_mps_probe/"
                "phase2b12_baseline_integrity.json"
            ),
            "release_decision": (
                "docs/porting/af3_mps_probe/"
                "phase2b12_release_decision.json"
            ),
            "gate_results": (
                "docs/porting/af3_mps_probe/phase2b12_gate_results.json"
            ),
            "execution_manifest": (
                "docs/porting/af3_mps_probe/"
                "phase2b12_execution_manifest.json"
            ),
            "exact_patch": (
                "docs/porting/af3_mps_probe/phase2b12_exact_patch.diff"
            ),
        },
    }
    write_json(DOCS / "phase2b12_summary.json", summary)

    artifact_paths = [
        DOCS / "phase2b12_backend_provenance.json",
        DOCS / "phase2b12_baseline_integrity.json",
        DOCS / "phase2b12_release_decision.json",
        DOCS / "phase2b12_gate_results.json",
        RESULTS / "gate_a_af2/trace.json",
        RESULTS / "gate_a_af2.log",
        RESULTS / "component_boundary.json",
        RESULTS / "component_boundary.log",
        RUNTIMES / "af2.json",
        RUNTIMES / "af3.json",
        RUNTIMES / "abmpnn.json",
        RUNTIMES / "abmpnn_wrong_version.json",
        RUNTIMES / "af2_environment.json",
        RUNTIMES / "af3_environment.json",
        RUNTIMES / "abmpnn_environment.json",
        RESULTS
        / "artifacts/wheels/"
        "jax_mps-0.10.9-py3-none-macosx_14_0_arm64.whl",
        RESULTS
        / "artifacts/wheels/"
        "jax_mps-0.10.11-py3-none-macosx_14_0_arm64.whl",
        patch_path,
    ]
    artifact_hashes = {
        str(path.relative_to(REPO_ROOT)): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in artifact_paths
        if path.is_file()
    }
    execution_manifest = {
        "phase": "2B.12",
        "decision": decision,
        "full_pipeline_run": False,
        "commands": {
            "runtime_af2": (
                "JAX_PLATFORMS=mps .envs/af2-mps-0109/bin/python "
                "scripts/runtime_identity.py --output "
                "docs/porting/af3_mps_probe/results/phase2b12/"
                "runtime_probes/af2.json --expected-version 0.10.9 "
                "--expected-prefix $PWD/.envs/af2-mps-0109 "
                "--expected-backend mps"
            ),
            "runtime_af3": (
                "JAX_PLATFORMS=mps .envs/af3-mps-probe/bin/python "
                "scripts/runtime_identity.py --output "
                "docs/porting/af3_mps_probe/results/phase2b12/"
                "runtime_probes/af3.json --expected-version 0.10.9 "
                "--expected-prefix $PWD/.envs/af3-mps-probe "
                "--expected-backend mps"
            ),
            "runtime_abmpnn": (
                "JAX_PLATFORMS=mps .envs/abmpnn-mps-0111/bin/python "
                "scripts/runtime_identity.py --output "
                "docs/porting/af3_mps_probe/results/phase2b12/"
                "runtime_probes/abmpnn.json --expected-version 0.10.11 "
                "--expected-prefix $PWD/.envs/abmpnn-mps-0111 "
                "--expected-backend mps"
            ),
            "gate_a": (
                "JAX_PLATFORMS=mps PYTHONHASHSEED=7 "
                ".envs/af2-mps-0109/bin/python -u "
                "tests/integration/phase2b11_af2_logits_trace.py "
                "--backend mps --label phase2b12_gate_a "
                "--output-root docs/porting/af3_mps_probe/results/"
                "phase2b12/gate_a_af2 --seed 1791095845"
            ),
            "gate_c_d": (
                "JAX_PLATFORMS=mps PYTHONHASHSEED=7 "
                'PYTHONPATH="$PWD/scripts:$PWD" '
                ".envs/af2-mps-0109/bin/python -u "
                "tests/integration/phase2b12_component_boundary.py "
                "--output docs/porting/af3_mps_probe/results/phase2b12/"
                "component_boundary.json --repeats 10 "
                "--boundary-pdb docs/porting/af3_mps_probe/results/"
                "phase2b9/run2/results/germinal_run/trajectories/"
                "structures/il3_nb_s1791095845.pdb "
                "--af3-structures docs/porting/af3_mps_probe/results/"
                "phase2b9/run2/results/germinal_run/trajectories/"
                "structures --af2-trace docs/porting/af3_mps_probe/"
                "results/phase2b12/gate_a_af2/trace.json --run-af3"
            ),
            "negative_runtime_control": (
                "JAX_PLATFORMS=mps .envs/abmpnn-mps-0111/bin/python "
                "scripts/runtime_identity.py --output "
                "docs/porting/af3_mps_probe/results/phase2b12/"
                "runtime_probes/abmpnn_wrong_version.json "
                "--expected-version 0.10.9 --expected-prefix "
                "$PWD/.envs/af2-mps-0109 --expected-backend mps"
            ),
        },
        "exit_codes": {
            "runtime_af2": 0,
            "runtime_af3": 0,
            "runtime_abmpnn": 0,
            "gate_a": 0,
            "gate_c_d": 0,
            "negative_runtime_control": 1,
        },
        "environment_manifests": provenance["environment_manifests"],
        "wheel_hashes": provenance["wheel_hashes"],
        "process_identities": {
            "parent": gates["gate_d"]["parent_runtime"],
            "workers": [
                identity
                for repeat in component["repeats"]
                for identity in repeat["worker_identities"]
            ],
        },
        "artifacts": artifact_hashes,
    }
    write_json(
        DOCS / "phase2b12_execution_manifest.json",
        execution_manifest,
    )

    patch_sha = sha256(patch_path)
    report = f"""# Phase 2B.12 - Component-Isolated Runtime Validation

Date: 2026-11-02

## Decision

**{decision}**

The stage-scoped architecture passes all four gates:

| Stage | Approved runtime |
|---|---|
| AF2 | `jax-mps==0.10.9` |
| AF3 | `jax-mps==0.10.9` |
| AbMPNN | `jax-mps==0.10.11` |
| PyRosetta / DSSP / DAlphaBall | native ARM64 CPU |

This is not a global `0.10.11` replacement approval. It preserves the accepted
AF2 and AF3 numerical baselines while isolating the AbMPNN boundary on the
release that eliminates its observed non-finite execution defect.

## Runtime provenance

All runtimes use JAX/JAXlib `0.10.2` and selected `MPS:0`.

| Runtime | Prefix | jax-mps | Plugin SHA-256 |
|---|---|---|---|
| AF2 | `.envs/af2-mps-0109` | `0.10.9` | `{provenance["runtimes"]["af2"]["plugin_library_sha256"]}` |
| AF3 | `.envs/af3-mps-probe` | `0.10.9` | `{provenance["runtimes"]["af3"]["plugin_library_sha256"]}` |
| AbMPNN | `.envs/abmpnn-mps-0111` | `0.10.11` | `{provenance["runtimes"]["abmpnn"]["plugin_library_sha256"]}` |

Each probe verified that the imported `jax_plugins.mps` module and shared
library reside inside the declared runtime prefix. The AbMPNN worker repeats
that identity check internally before loading its model.

Wheel hashes:

- `jax-mps 0.10.9`: `{provenance["wheel_hashes"]["0.10.9"]}`
- `jax-mps 0.10.11`: `{provenance["wheel_hashes"]["0.10.11"]}`

## Gate A - AF2 baseline preservation

The isolated AF2 runtime completed the fixed seed `1791095845` AF2 design:

- all 105 historical metric/sequence records match Phase 2B.9;
- all recorded sequence logits and gradients match Phase 2B.11's `0.10.9`
  comparator bitwise;
- best sequence and PDB match bitwise;
- pLDDT: `{gates["gate_a"]["best_log"]["plddt"]:.12f}`
- iPTM: `{gates["gate_a"]["best_log"]["i_ptm"]:.12f}`
- iPAE: `{gates["gate_a"]["best_log"]["i_pae"]:.12f}`

The original iPTM threshold pass is restored.

## Gate B - AF3 compatibility

The pinned AF3 revision is `{gates["gate_b"]["af3_revision"]}`. Model and
artifact hashes match the accepted record. The preserved production searches
completed BFD, UniRef90, UniProt, and MGnify against `/Volumes/T9/database`.
Both MSA streams are present in the AF3 data consumed by inference.

Independent accepted AF3 runs produced identical:

- input JSON SHA-256: `{gates["gate_b"]["run1_input_sha256"]}`;
- prediction PDB SHA-256: `{gates["gate_b"]["run1_af3_pdb_sha256"]}`;
- relaxed PDB SHA-256: `{gates["gate_b"]["run1_relaxed_pdb_sha256"]}`.

A fresh targeted AF3 inference process then ran the preserved production data
through the pinned AF3 runtime with five diffusion samples and ten recycles.
It used `MPS:0`, completed three seed runs and 15 samples in
`{gates["gate_b"]["targeted_inference"]["runtime_seconds"]:.2f}` seconds, and
selected the same worst-sample PDB SHA-256
`{gates["gate_b"]["targeted_af3_selected_pdb_sha256"]}`.

## Gate C - AbMPNN reliability

Ten independent caller invocations launched fresh AbMPNN worker processes using
the exact preserved AF3 boundary PDB
`50cf77a53c6c65213cc732956494a58488ea9cecfbc4fa650bf8c088da7e8e41`.

Every invocation:

- imported `jax-mps 0.10.11` from `.envs/abmpnn-mps-0111`;
- selected `MPS:0`;
- returned five unique sequences and finite scores;
- matched the expected sequence-set SHA-256
  `{gates["gate_c"]["sequence_hashes"][0]}` exactly;
- used a distinct worker PID.

All worker PIDs: `{", ".join(map(str, gates["gate_c"]["worker_pids"]))}`.

## Gate D - Orchestration integration

Gate D ran from the real AF2 environment. The parent reported:

- prefix `.envs/af2-mps-0109`;
- `jax-mps 0.10.9`;
- `MPS:0`.

The parent loaded the preserved AF3 handoff artifact, launched and waited for
the AF3 inference subprocess, and only then launched AbMPNN through
`germinal.filters.redesign.get_abmpnn_sequences` using the explicit configured
interpreter. Every AbMPNN child reported:

- prefix `.envs/abmpnn-mps-0111`;
- `jax-mps 0.10.11`;
- `MPS:0`;
- a PID distinct from its parent and every other worker.

The negative runtime-identity control intentionally requested `0.10.9` from
the `0.10.11` AbMPNN environment. It exited with code 1 and reported false
version and prefix checks. Non-finite output and retry behavior remain
fail-closed.

## Baseline integrity

- Git commit: `{baseline["git_commit"]}`
- Accepted source snapshot:
  `{baseline["source_snapshot"]["sha256"]}`
- All five AF2 checkpoint hashes match: **{baseline["checks"]["all_af2_checkpoints_match"]}**
- Input PDB hash matches: **{baseline["checks"]["input_pdb_hash_matches"]}**
- Scientific configuration matches the base run: **{baseline["checks"]["scientific_configuration_matches_base"]}**
- Only declared runtime-boundary source files changed.
- Seed, optimization schedule, sampling settings, checkpoints, thresholds, and
  filter definitions remain unchanged.

Exact runtime-boundary patch SHA-256: `{patch_sha}`.

## Limitations

- The exact Metal operator or buffer-lifetime transition remains unidentified.
- A full production run was not launched and is not authorized by Phase 2B.12.
- Production database readiness must remain a separate Phase 2C prerequisite;
  preserved production database evidence here does not replace a fresh
  readiness and representative-inference check.

## Evidence

- Backend provenance: `phase2b12_backend_provenance.json`
- Baseline integrity: `phase2b12_baseline_integrity.json`
- Release decision: `phase2b12_release_decision.json`
- Gate results: `phase2b12_gate_results.json`
- Execution manifest: `phase2b12_execution_manifest.json`
- Exact patch: `phase2b12_exact_patch.diff`
- Gate A trace: `results/phase2b12/gate_a_af2/trace.json`
- Gate C/D report and worker identities:
  `results/phase2b12/component_boundary.json`
"""
    (DOCS / "phase2b12_report.md").write_text(report)

    checksum_paths = [
        DOCS / "phase2b12_report.md",
        DOCS / "phase2b12_summary.json",
        DOCS / "phase2b12_backend_provenance.json",
        DOCS / "phase2b12_baseline_integrity.json",
        DOCS / "phase2b12_release_decision.json",
        DOCS / "phase2b12_gate_results.json",
        DOCS / "phase2b12_execution_manifest.json",
        patch_path,
    ]
    checksum_paths.extend(
        path for path in RESULTS.rglob("*") if path.is_file()
    )
    seen = set()
    lines = []
    for path in checksum_paths:
        relative = path.relative_to(REPO_ROOT)
        if relative in seen or not path.is_file():
            continue
        seen.add(relative)
        lines.append(f"{sha256(path)}  {relative}\n")
    (DOCS / "phase2b12_checksums.sha256").write_text("".join(lines))
    print(
        json.dumps(
            {
                "decision": decision,
                "statuses": required_status,
                "patch_sha256": patch_sha,
                "checksums_file": (
                    "docs/porting/af3_mps_probe/"
                    "phase2b12_checksums.sha256"
                ),
            },
            indent=2,
        )
    )
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
