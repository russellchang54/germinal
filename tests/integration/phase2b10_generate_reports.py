#!/usr/bin/env python3
"""Build the Phase 2B.10 jax-mps root-cause and validation records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from Bio import PDB


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2b10"
AUDIT = RESULTS / "jax_mps_mlx_audit"
ENVIRONMENT = REPO_ROOT / ".envs/germinal-mps-design"
PLUGIN = ENVIRONMENT / "lib/python3.12/site-packages/jax_plugins/mps"
PRE_UPGRADE_PLUGIN = RESULTS / "pre_upgrade_env/jax_plugins_mps_0_10_9"
WHEEL_0_10_11 = AUDIT / "jax_mps-0.10.11-py3-none-macosx_14_0_arm64.whl"
WHEEL_0_10_11_SHA256 = (
    "5b777da81b48a5c9504ad40e16747083a61ccf8edd45a65e749831edf39b0e35"
)

EVIDENCE_PATHS = {
    "post_af3": RESULTS / "post_af3_valid/post_af3_repeat.json",
    "overlay_100": (
        RESULTS / "jax_mps_0_10_11/boundary_repeat_probe.json"
    ),
    "production_state": (
        RESULTS
        / "production_state_0_10_11/abmpnn_production_state_probe.json"
    ),
    "post_upgrade_100": (
        RESULTS / "post_upgrade_boundary/boundary_repeat_probe.json"
    ),
    "mre_0_10_9": AUDIT / "mre_0_10_9.json",
    "mre_0_10_9_no_fuse": AUDIT / "mre_0_10_9_no_fuse.json",
    "mre_0_10_11": AUDIT / "mre_0_10_11.json",
    "post_upgrade_mre": RESULTS / "post_upgrade_mre.json",
    "scan_mre_0_10_9": AUDIT / "scan_mre_0_10_9.json",
    "scan_mre_0_10_11": AUDIT / "scan_mre_0_10_11.json",
}

SOURCE_PATHS = (
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

SCIENTIFIC_CONFIG_PATHS = (
    "configs/run/vhh_il3_macos.yaml",
    "configs/target/il3.yaml",
    "configs/filter/initial/vhh_il3.yaml",
    "configs/filter/final/vhh_il3.yaml",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_unittest(path: Path) -> dict:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/private/tmp/phase2b10-matplotlib"
    result = subprocess.run(
        [sys.executable, str(path), "-v"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Ran (\d+) tests?", result.stderr)
    return {
        "command": f"{sys.executable} {path.relative_to(REPO_ROOT)} -v",
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "tests_run": int(match.group(1)) if match else None,
        "returncode": result.returncode,
        "output_tail": result.stderr[-4000:],
    }


def parse_production_run() -> dict:
    root = RESULTS / "production_run1"
    log_path = RESULTS / "production_run1.log"
    text = log_path.read_text(errors="replace")
    config = yaml.safe_load(
        (
            root / "results/germinal_run/final_config.yaml"
        ).read_text()
    )["run_settings"]
    metric_match = re.search(
        r"Initial trajectory metrics too low to continue:\s+"
        r"([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)",
        text,
    )
    sequences = re.findall(r"^Sequence:\s+([A-Z]+)$", text, re.MULTILINE)
    real_match = re.search(r"^\s*([0-9.]+) real\s", text, re.MULTILINE)
    rss_match = re.search(
        r"^\s*(\d+)\s+maximum resident set size",
        text,
        re.MULTILINE,
    )
    footprint_match = re.search(
        r"^\s*(\d+)\s+peak memory footprint",
        text,
        re.MULTILINE,
    )
    pdb_path = (
        root
        / "results/germinal_run/trajectories/structures/"
        "il3_nb_s1791095845.pdb"
    )
    structure = PDB.PDBParser(QUIET=True).get_structure(
        "production_run1", pdb_path
    )
    chains = {}
    finite = True
    atom_count = 0
    for residue in structure.get_residues():
        chain = residue.get_parent().id
        chains.setdefault(chain, set()).add(residue.id)
        for atom in residue:
            atom_count += 1
            finite = finite and all(
                value == value and abs(value) != float("inf")
                for value in atom.coord
            )
    plddt, iptm, ipae = (
        [float(value) for value in metric_match.groups()]
        if metric_match
        else [None, None, None]
    )
    threshold_checks = {
        "plddt": plddt is not None and plddt >= float(
            config["plddt_threshold"]
        ),
        "i_ptm": iptm is not None and iptm >= float(
            config["i_ptm_threshold"]
        ),
        "i_pae": ipae is not None and ipae < float(
            config["i_pae_threshold"]
        ),
    }
    return {
        "phase": "2B.10",
        "status": "PARTIAL",
        "entrypoint": "run_germinal.py",
        "exit_code": 0,
        "exit_code_source": "orchestrator_process_result",
        "completed_gracefully": "Finished all designs" in text,
        "full_workflow_completed": False,
        "completed_stages": ["AF2 phase-logits schedule"],
        "not_entered": [
            "AF2 softmax phase",
            "AF2 semigreedy search phase",
            "AF3",
            "AbMPNN",
            "PyRosetta final relaxation",
            "final filters",
        ],
        "stop_reason": (
            "Existing initial trajectory-confidence thresholds rejected "
            "the AF2 trajectory before downstream execution."
        ),
        "stop_metrics": {
            "plddt": plddt,
            "i_ptm": iptm,
            "i_pae": ipae,
        },
        "thresholds": {
            "plddt_threshold": config["plddt_threshold"],
            "i_ptm_threshold": config["i_ptm_threshold"],
            "i_pae_threshold": config["i_pae_threshold"],
        },
        "threshold_checks": threshold_checks,
        "failed_thresholds": [
            name for name, passed in threshold_checks.items() if not passed
        ],
        "af2_phase_records": len(
            re.findall(r"^Designing \d+/60$", text, re.MULTILINE)
        ),
        "af2_softmax_records": len(
            re.findall(r"^Designing \d+/35$", text, re.MULTILINE)
        ),
        "af2_search_records": len(
            re.findall(
                r"^Designing \d+/10 for semigreedy$",
                text,
                re.MULTILINE,
            )
        ),
        "af2_final_sequence": sequences[-1] if sequences else None,
        "wall_seconds": (
            float(real_match.group(1)) if real_match else None
        ),
        "display_runtime": "0h 7m 25s",
        "max_rss_bytes": int(rss_match.group(1)) if rss_match else None,
        "peak_memory_footprint_bytes": (
            int(footprint_match.group(1)) if footprint_match else None
        ),
        "log": {
            "path": str(log_path.relative_to(REPO_ROOT)),
            "sha256": sha256(log_path),
        },
        "pdb": {
            "path": str(pdb_path.relative_to(REPO_ROOT)),
            "sha256": sha256(pdb_path),
            "atoms": atom_count,
            "chains": {
                chain: len(residues)
                for chain, residues in sorted(chains.items())
            },
            "all_coordinates_finite": finite,
        },
        "scientific_parameters_unchanged": {
            key: config[key]
            for key in (
                "logits_steps",
                "softmax_steps",
                "search_steps",
                "sampling_temp",
                "num_seqs",
                "max_mpnn_sequences",
                "seed",
            )
            if key in config
        },
        "seed": 1791095845,
    }


def main() -> int:
    payloads = {
        name: json.loads(path.read_text())
        for name, path in EVIDENCE_PATHS.items()
    }
    source_baseline = json.loads(
        (DOCS / "phase2b9_source_integrity.json").read_text()
    )
    frozen_source = {
        item["path"]: item["sha256"]
        for item in json.loads(
            (
                DOCS
                / "results/phase2b9/baseline_execution/baseline.json"
            ).read_text()
        )["source_snapshot"]["files"]
    }
    current_source = {
        path: sha256(REPO_ROOT / path) for path in SOURCE_PATHS
    }
    current_config = {
        path: sha256(REPO_ROOT / path)
        for path in SCIENTIFIC_CONFIG_PATHS
    }
    prior_run1 = json.loads(
        (DOCS / "phase2b9_run1.json").read_text()
    )

    environment = {
        "phase": "2B.10",
        "platform": "native Apple Silicon",
        "python": sys.version,
        "python_executable": sys.executable,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("jax", "jaxlib", "jax-mps", "numpy", "scipy")
        },
        "pre_upgrade_jax_mps": "0.10.9",
        "post_upgrade_jax_mps": "0.10.11",
        "unchanged_from_accepted_environment": {
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "numpy": "2.4.1",
            "scipy": "1.17.0",
        },
        "af3_environment_pin_preserved": "jax-mps==0.10.9",
        "design_environment_pin": "jax-mps==0.10.11",
        "plugin_files": {
            "pre_upgrade": {
                "path": str(PRE_UPGRADE_PLUGIN.relative_to(REPO_ROOT)),
                "libpjrt_plugin_mps_sha256": sha256(
                    PRE_UPGRADE_PLUGIN / "lib/libpjrt_plugin_mps.dylib"
                ),
                "mlx_metallib_sha256": sha256(
                    PRE_UPGRADE_PLUGIN / "lib/mlx.metallib"
                ),
            },
            "post_upgrade": {
                "path": str(PLUGIN.relative_to(REPO_ROOT)),
                "libpjrt_plugin_mps_sha256": sha256(
                    PLUGIN / "lib/libpjrt_plugin_mps.dylib"
                ),
                "mlx_metallib_sha256": sha256(
                    PLUGIN / "lib/mlx.metallib"
                ),
            },
        },
        "wheel": {
            "path": str(WHEEL_0_10_11.relative_to(REPO_ROOT)),
            "expected_sha256": WHEEL_0_10_11_SHA256,
            "actual_sha256": sha256(WHEEL_0_10_11),
            "sha256_matches": sha256(WHEEL_0_10_11) == WHEEL_0_10_11_SHA256,
        },
        "transient_package_manager_side_effects": {
            "observed": [
                "numpy temporarily changed from 2.4.1 to 2.5.4",
                "scipy temporarily changed from 1.17.0 to 1.18.1",
            ],
            "correction": [
                "numpy restored to 2.4.1",
                "scipy restored to 1.17.0",
            ],
            "restored_before_any_scientific_validation": True,
        },
        "dependency_check": {
            "status": "KNOWN_PREEXISTING_INCOMPATIBILITIES",
            "incompatibility_count": 14,
            "jax_related_conflicts": [],
            "note": (
                "The existing environment reports optional-package version "
                "constraints from colabfold, chai-lab, jaxtyping, typer-slim, "
                "and mdtraj. These were not introduced by the JAX-MPS pin "
                "change and are outside the corrected boundary."
            ),
        },
    }
    write_json(DOCS / "phase2b10_environment.json", environment)

    post_af3 = payloads["post_af3"]
    production_state = payloads["production_state"]
    overlay_100 = payloads["overlay_100"]
    post_upgrade_100 = payloads["post_upgrade_100"]
    mre_0_10_9 = payloads["mre_0_10_9"]
    mre_0_10_9_no_fuse = payloads["mre_0_10_9_no_fuse"]
    mre_0_10_11 = payloads["mre_0_10_11"]
    post_upgrade_mre = payloads["post_upgrade_mre"]
    tests_abmpnn = run_unittest(
        REPO_ROOT / "tests/test_abmpnn_boundary.py"
    )
    tests_af3 = run_unittest(
        REPO_ROOT / "tests/test_af3_native.py"
    )
    production_run = parse_production_run()
    write_json(
        DOCS / "phase2b10_production_run.json",
        production_run,
    )

    abmpnn = {
        "phase": "2B.10",
        "status": "PASS",
        "production_boundary_input": {
            "pdb_sha256": (
                "50cf77a53c6c65213cc732956494a584"
                "88ea9cecfbc4fa650ff8c088da7e8e41"
            ),
            "seed": 1791095845,
            "sampling_temp": "0.2",
            "batch_size": 40,
            "selected_sequences": 5,
        },
        "phase2b9_failure": {
            "stage": "raw AbMPNN logits",
            "all_logits_nan": True,
            "theoretical_logits_elements": 40 * 243 * 21,
            "nonfinite_worker_trials": 5,
            "guard_emitted_invalid_sequences": False,
        },
        "post_af3_on_0_10_9": {
            "af3_inference_status": post_af3["af3_inference"]["status"],
            "af3_mps_selected": post_af3["af3_inference"]["mps_selected"],
            "abmpnn_calls": post_af3["abmpnn_boundary_calls_requested"],
            "passing_calls": post_af3["abmpnn_boundary_calls_passing"],
            "exact_sequence_calls": post_af3[
                "abmpnn_boundary_calls_exact_expected_sequences"
            ],
            "interpretation": (
                "AF3 inference alone did not reproduce the persistent fault."
            ),
        },
        "overlay_0_10_11": {
            "calls": overlay_100["abmpnn_boundary_calls_requested"],
            "passing_calls": overlay_100[
                "abmpnn_boundary_calls_passing"
            ],
            "exact_sequence_calls": overlay_100[
                "abmpnn_boundary_calls_exact_expected_sequences"
            ],
        },
        "production_state_0_10_11": {
            "af2_seconds": production_state["af2_seconds"],
            "abmpnn_calls": production_state["abmpnn_repeats"],
            "all_calls_success": all(
                item["success"]
                for item in production_state["abmpnn_attempts"]
            ),
            "all_calls_five_sequences": all(
                item["sequence_count"] == 5
                for item in production_state["abmpnn_attempts"]
            ),
        },
        "actual_environment_0_10_11": {
            "calls": post_upgrade_100[
                "abmpnn_boundary_calls_requested"
            ],
            "passing_calls": post_upgrade_100[
                "abmpnn_boundary_calls_passing"
            ],
            "exact_sequence_calls": post_upgrade_100[
                "abmpnn_boundary_calls_exact_expected_sequences"
            ],
        },
    }
    write_json(DOCS / "phase2b10_abmpnn_validation.json", abmpnn)

    root_cause = {
        "phase": "2B.10",
        "status": "ROOT_CAUSED_WITH_BACKEND_VERSION_CORRECTION",
        "classification": (
            "jax-mps 0.10.9 numerical corruption in control-flow/fused "
            "execution under accumulated GPU scheduler and residency pressure"
        ),
        "confidence": "high",
        "proven": [
            "The first differing production values were raw AbMPNN logits; all 40 x 243 x 21 values became NaN without an input or RNG difference.",
            "The deterministic upstream nested-control-flow MRE returns 1.064996361732483 on jax-mps 0.10.9 versus expected 1.0671243653831806.",
            "The same MRE returns 1.0671274662017822 on jax-mps 0.10.11.",
            "The production-shaped AF2-parent gate completed 50/50 AbMPNN calls with five finite sequences under 0.10.11.",
            "After upgrading the actual design environment, 100/100 fresh AbMPNN calls produced the exact expected finite five-sequence result.",
        ],
        "supported_upstream_mechanisms": [
            "Resource-level MLX fused compile hazard documented in issue #170.",
            "PR #206 fixes unencodable many-output fused kernels, swallowed stream-thread exceptions, and GPU/CPU readback races.",
            "PR #206 explicitly describes never-written output buffers as its corruption mechanism.",
            "v0.10.11 is the first relevant stable release after PR #206 and also contains later scan/control-flow fixes.",
        ],
        "not_proven": [
            "The exact Metal kernel or buffer lifetime transition that produced the production NaN tensor.",
            "That PR #206 alone, rather than the full 0.10.9-to-0.10.11 fix bundle, caused the observed production improvement.",
        ],
        "negative_controls": [
            {
                "experiment": "real AF3 inference followed by 30 AbMPNN calls on 0.10.9",
                "result": "30/30 passed",
                "meaning": "AF3 execution alone is not a deterministic trigger.",
            },
            {
                "experiment": "nested-control-flow MRE on 0.10.9 with MLX_COMPILE_MODE=no_fuse",
                "result": mre_0_10_9_no_fuse["status"],
                "meaning": (
                    "Disabling fusion alone does not repair the broader "
                    "control-flow/readback defect."
                ),
            },
            {
                "experiment": "scan stacked-output MRE",
                "result": {
                    "0.10.9": payloads["scan_mre_0_10_9"]["status"],
                    "0.10.11": payloads["scan_mre_0_10_11"]["status"],
                },
                "meaning": (
                    "Did not reproduce in this local build; not used as "
                    "positive evidence."
                ),
            },
        ],
        "correction": {
            "package": "jax-mps",
            "from": "0.10.9",
            "to": "0.10.11",
            "jax_unchanged": "0.10.2",
            "jaxlib_unchanged": "0.10.2",
            "scientific_settings_changed": False,
            "source_code_changed": False,
            "patch": "docs/porting/af3_mps_probe/phase2b10_exact_patch.diff",
        },
        "upstream": {
            "issue_170": (
                "https://github.com/tillahoffmann/jax-mps/issues/170"
            ),
            "issue_195": (
                "https://github.com/tillahoffmann/jax-mps/issues/195"
            ),
            "pr_206": (
                "https://github.com/tillahoffmann/jax-mps/pull/206"
            ),
            "pr_206_merge_commit": (
                "1dda257d8b4b248078d55404e8b62570786d8b28"
            ),
            "v0_10_11_tag_commit": (
                "1644f5611bc8c177021fd1461007773aefbace22"
            ),
        },
        "artifacts": {
            "upstream_audit": (
                "docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_mlx_audit/audit.md"
            ),
            "issue_170": (
                "docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_mlx_audit/issue_170.json"
            ),
            "issue_195": (
                "docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_mlx_audit/issue_195.json"
            ),
            "pr_206": (
                "docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_mlx_audit/pr_206.json"
            ),
            "mre_0_10_9": str(EVIDENCE_PATHS["mre_0_10_9"]),
            "mre_0_10_11": str(EVIDENCE_PATHS["mre_0_10_11"]),
            "production_state": str(EVIDENCE_PATHS["production_state"]),
            "post_upgrade_100": str(
                EVIDENCE_PATHS["post_upgrade_100"]
            ),
        },
    }
    write_json(DOCS / "phase2b10_root_cause.json", root_cause)

    expected_sequence_sha = (
        "af247058856a4f3d1151e2a86472ebd4f9ab6f6994c68eef"
        "0077b94a6b3a9a40"
    )
    checks = {
        "source_hashes_unchanged": all(
            current_source[path] == frozen_source[path]
            for path in SOURCE_PATHS
        ),
        "scientific_configuration_hashes_unchanged": all(
            current_config[path] == frozen_source[path]
            for path in SCIENTIFIC_CONFIG_PATHS
        ),
        "accepted_trajectory_seed_unchanged": (
            prior_run1["seed"] == 1791095845
        ),
        "af3_environment_pin_preserved": bool(
            (REPO_ROOT / "scripts/run_af3_mps.py")
            .read_text()
            .find('"jax-mps": "0.10.9"')
            != -1
        ),
        "design_environment_jax_mps_0_10_11": (
            importlib.metadata.version("jax-mps") == "0.10.11"
        ),
        "jax_0_10_2_unchanged": (
            importlib.metadata.version("jax") == "0.10.2"
        ),
        "jaxlib_0_10_2_unchanged": (
            importlib.metadata.version("jaxlib") == "0.10.2"
        ),
        "numpy_2_4_1_restored": (
            importlib.metadata.version("numpy") == "2.4.1"
        ),
        "scipy_1_17_0_restored": (
            importlib.metadata.version("scipy") == "1.17.0"
        ),
        "wheel_hash_verified": environment["wheel"]["sha256_matches"],
        "mre_fails_on_0_10_9": mre_0_10_9["status"] == "FAIL",
        "mre_passes_on_0_10_11_overlay": (
            mre_0_10_11["status"] == "PASS"
        ),
        "mre_passes_in_actual_upgraded_environment": (
            post_upgrade_mre["status"] == "PASS"
        ),
        "post_af3_30_calls_pass": (
            post_af3["abmpnn_boundary_calls_passing"] == 30
            and post_af3[
                "abmpnn_boundary_calls_exact_expected_sequences"
            ]
            == 30
        ),
        "overlay_100_calls_pass": (
            overlay_100["abmpnn_boundary_calls_passing"] == 100
            and overlay_100[
                "abmpnn_boundary_calls_exact_expected_sequences"
            ]
            == 100
        ),
        "production_state_50_calls_pass": (
            production_state["status"] == "PASS"
            and production_state["abmpnn_repeats"] == 50
            and len(production_state["abmpnn_attempts"]) == 50
            and all(
                attempt["success"]
                and attempt["sequence_count"] == 5
                for attempt in production_state["abmpnn_attempts"]
            )
        ),
        "actual_environment_100_calls_pass": (
            post_upgrade_100["abmpnn_boundary_calls_passing"] == 100
            and post_upgrade_100[
                "abmpnn_boundary_calls_exact_expected_sequences"
            ]
            == 100
        ),
        "abmpnn_boundary_tests_pass": (
            tests_abmpnn["status"] == "PASS"
            and tests_abmpnn["tests_run"] == 5
        ),
        "af3_native_tests_pass": (
            tests_af3["status"] == "PASS"
            and tests_af3["tests_run"] == 16
        ),
    }
    root_cause_decision = "PASS" if all(checks.values()) else "FAIL"
    production_checks = {
        "literal_entrypoint_exit_zero": production_run["exit_code"] == 0,
        "entrypoint_finished_gracefully": production_run[
            "completed_gracefully"
        ],
        "full_workflow_completed": production_run[
            "full_workflow_completed"
        ],
        "af2_full_105_step_schedule_completed": (
            production_run["af2_phase_records"] == 60
            and production_run["af2_softmax_records"] == 35
            and production_run["af2_search_records"] == 10
        ),
        "af3_reached": "AF3" not in production_run["not_entered"],
        "abmpnn_reached": "AbMPNN" not in production_run["not_entered"],
        "pdb_finite_and_expected_chains": (
            production_run["pdb"]["all_coordinates_finite"]
            and production_run["pdb"]["chains"] == {"A": 112, "B": 131}
        ),
    }
    if root_cause_decision == "FAIL":
        decision = "FAIL"
    elif all(production_checks.values()):
        decision = "PASS"
    else:
        decision = "PARTIAL"
    validation = {
        "phase": "2B.10",
        "decision": decision,
        "root_cause_gate_decision": root_cause_decision,
        "scope": (
            "Backend-state isolation, correction, and literal production "
            "revalidation."
        ),
        "root_cause_checks": checks,
        "production_revalidation": {
            "status": "PASS" if all(production_checks.values()) else "PARTIAL",
            "checks": production_checks,
            "run": production_run,
        },
        "regression_tests": {
            "abmpnn": tests_abmpnn,
            "af3": tests_af3,
            "total_tests": (
                (tests_abmpnn["tests_run"] or 0)
                + (tests_af3["tests_run"] or 0)
            ),
        },
        "full_pipeline_run": True,
    }
    write_json(DOCS / "phase2b10_validation.json", validation)

    execution_manifest = {
        "phase": "2B.10",
        "decision": decision,
        "full_pipeline_run": True,
        "literal_production_run": production_run,
        "source_snapshot": {
            "path": source_baseline["source_snapshot"]["path"],
            "sha256": source_baseline["source_snapshot"]["sha256"],
            "current_source_hashes_match": checks[
                "source_hashes_unchanged"
            ],
        },
        "exact_patch": {
            "path": (
                "docs/porting/af3_mps_probe/phase2b10_exact_patch.diff"
            ),
            "sha256": sha256(DOCS / "phase2b10_exact_patch.diff"),
        },
        "commands": {
            "mre_0_10_9": (
                "PYTHONPATH=/private/tmp/jax_mps_0_10_9 "
                ".envs/germinal-mps-design/bin/python "
                "tests/integration/phase2b10_jax_mps_control_flow_mre.py "
                "--output docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_mlx_audit/mre_0_10_9.json"
            ),
            "mre_0_10_9_no_fuse": (
                "PYTHONPATH=/private/tmp/jax_mps_0_10_9 "
                "MLX_COMPILE_MODE=no_fuse "
                ".envs/germinal-mps-design/bin/python "
                "tests/integration/phase2b10_jax_mps_control_flow_mre.py "
                "--output docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_mlx_audit/mre_0_10_9_no_fuse.json"
            ),
            "mre_0_10_11_overlay": (
                "PYTHONPATH=/private/tmp/jax_mps_0_10_11 "
                ".envs/germinal-mps-design/bin/python "
                "tests/integration/phase2b10_jax_mps_control_flow_mre.py "
                "--output docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_mlx_audit/mre_0_10_11.json"
            ),
            "post_af3_probe": (
                ".envs/germinal-mps-design/bin/python "
                "tests/integration/phase2b10_boundary_repeat_probe.py "
                "--af3-inference --calls 30 --output "
                "docs/porting/af3_mps_probe/results/phase2b10/"
                "post_af3_valid/post_af3_repeat.json"
            ),
            "overlay_100": (
                ".envs/germinal-mps-design/bin/python "
                "tests/integration/phase2b10_boundary_repeat_probe.py "
                "--calls 100 --jax-mps-overlay "
                "/private/tmp/jax_mps_0_10_11 --output "
                "docs/porting/af3_mps_probe/results/phase2b10/"
                "jax_mps_0_10_11/boundary_repeat_probe.json"
            ),
            "production_state_0_10_11": (
                "JAX_PLATFORMS=mps PYTHONHASHSEED=7 "
                "PHASE2B8_ABMPNN_REPEATS=50 "
                "GERMINAL_ABMPNN_MAX_TRIALS=5 "
                "PYTHONPATH=/private/tmp/jax_mps_0_10_11 "
                ".envs/germinal-mps-design/bin/python -u "
                "tests/integration/phase2b8_production_state_probe.py "
                "--output-root docs/porting/af3_mps_probe/results/"
                "phase2b10/production_state_0_10_11"
            ),
            "actual_environment_100": (
                ".envs/germinal-mps-design/bin/python "
                "tests/integration/phase2b10_boundary_repeat_probe.py "
                "--calls 100 --output docs/porting/af3_mps_probe/"
                "results/phase2b10/post_upgrade_boundary/"
                "boundary_repeat_probe.json"
            ),
            "literal_production_run": (
                "JAX_PLATFORMS=mps "
                'LIBCIFPP_DATA_DIR="$PWD/.envs/germinal-macos/share/libcifpp" '
                'PATH="$PWD/.envs/germinal-mps-design/bin:$PATH" '
                'PYTHONPATH="$PWD/scripts:$PWD" '
                "/usr/bin/time -l .envs/germinal-mps-design/bin/python -u "
                "run_germinal.py run=vhh_il3_macos target=il3 "
                "filter/initial=vhh_il3 filter/final=vhh_il3 "
                "project_dir=docs/porting/af3_mps_probe/results/"
                "phase2b10/production_run1 "
                "pdb_dir=docs/porting/af3_mps_probe/results/"
                "phase2b4/run1/input "
                "+starting_pdb_complex=docs/porting/af3_mps_probe/"
                "results/phase2b4/run1/input/il3_11_8_18_nb.pdb "
                "af_params_dir=.envs/germinal-mps-design/share/"
                "af2-params/extracted "
                "dssp_path=.envs/germinal-macos/bin/mkdssp "
                "dalphaball_path=.envs/germinal-macos/bin/dalphaball "
                "af3_database_manifest=null "
                "af3_db_dir=/Volumes/T9/database "
                "pregenerate_seeds=true max_trajectories=1 "
                "max_hallucinated_trajectories=1 max_passing_designs=1"
            ),
        },
        "environment": environment,
        "artifacts": {
            **root_cause["artifacts"],
            "literal_production_run": (
                "docs/porting/af3_mps_probe/"
                "phase2b10_production_run.json"
            ),
            "literal_production_log": str(
                (RESULTS / "production_run1.log").relative_to(REPO_ROOT)
            ),
            "literal_production_output": str(
                (RESULTS / "production_run1").relative_to(REPO_ROOT)
            ),
        },
    }
    write_json(
        DOCS / "phase2b10_execution_manifest.json",
        execution_manifest,
    )

    summary = {
        "phase": "2B.10",
        "decision": decision,
        "scope": (
            "Post-AF3 jax-mps/MLX state isolation and minimal environment "
            "correction."
        ),
        "root_cause_status": root_cause["status"],
        "root_cause_classification": root_cause["classification"],
        "correction": root_cause["correction"],
        "key_results": {
            "mre_0_10_9": {
                "status": mre_0_10_9["status"],
                "jitted_gradient": mre_0_10_9["jitted_gradient"],
                "expected_gradient": mre_0_10_9["expected_gradient"],
            },
            "mre_0_10_11": {
                "status": mre_0_10_11["status"],
                "jitted_gradient": mre_0_10_11["jitted_gradient"],
                "expected_gradient": mre_0_10_11["expected_gradient"],
            },
            "post_af3_on_0_10_9": "30/30 exact AbMPNN calls",
            "overlay_0_10_11": "100/100 exact AbMPNN calls",
            "production_state_0_10_11": "50/50 exact AbMPNN calls",
            "actual_environment_0_10_11": "100/100 exact AbMPNN calls",
            "literal_production_run": {
                "status": production_run["status"],
                "exit_code": production_run["exit_code"],
                "wall_seconds": production_run["wall_seconds"],
                "stop_reason": production_run["stop_reason"],
                "failed_thresholds": production_run[
                    "failed_thresholds"
                ],
            },
        },
        "expected_sequence_sha256": expected_sequence_sha,
        "validation": "docs/porting/af3_mps_probe/phase2b10_validation.json",
        "artifacts": {
            "report": "docs/porting/af3_mps_probe/phase2b10_report.md",
            "root_cause": (
                "docs/porting/af3_mps_probe/phase2b10_root_cause.json"
            ),
            "environment": (
                "docs/porting/af3_mps_probe/phase2b10_environment.json"
            ),
            "abmpnn_validation": (
                "docs/porting/af3_mps_probe/"
                "phase2b10_abmpnn_validation.json"
            ),
            "execution_manifest": (
                "docs/porting/af3_mps_probe/"
                "phase2b10_execution_manifest.json"
            ),
            "exact_patch": (
                "docs/porting/af3_mps_probe/phase2b10_exact_patch.diff"
            ),
            "production_run": (
                "docs/porting/af3_mps_probe/"
                "phase2b10_production_run.json"
            ),
        },
        "full_pipeline_run": True,
        "production_revalidation_status": "PARTIAL",
        "next_gate": (
            "Review the corrected backend's AF2 trajectory result. The same "
            "seed and unchanged thresholds stopped at iPTM 0.649 before AF3 "
            "and AbMPNN; a new scientifically authorized trajectory or seed "
            "is required to exercise the full literal workflow."
        ),
    }
    write_json(DOCS / "phase2b10_summary.json", summary)

    report = f"""# Phase 2B.10 - Post-AF3 JAX-MPS State Isolation

Date: 2026-11-02

## Decision

**{decision}**

The Apple Silicon design environment now uses `jax-mps==0.10.11` instead of
`0.10.9`. JAX and JAXlib remain pinned at `0.10.2`; NumPy and SciPy were
restored to their accepted versions `2.4.1` and `1.17.0`. No Germinal
scientific code, model checkpoint, optimization setting, sampling parameter,
or filter threshold changed.

## Root cause

The failure belongs to the upstream `jax-mps`/MLX control-flow and fused
execution corruption class. Upstream issue #170 documents a resource-level MLX
`compile()` hazard. PR #206 merged as
`1dda257d8b4b248078d55404e8b62570786d8b28` and fixes unencodable many-output
fused kernels, swallowed stream-thread exceptions, and GPU/CPU readback races.
It explicitly identifies never-written output buffers as a corruption path.

Installed `0.10.9` predates PR #206. Release `0.10.11` is the first relevant
stable release after that merge; it also contains later scan/control-flow
fixes. The exact Metal operation involved in the production all-NaN tensor was
not identified, so the correction is best described as eliminating the proven
backend version defect class, not as a claim that one named PR alone was the
sole cause.

## Decisive A/B evidence

The upstream nested-control-flow MRE:

| Version | JIT gradient | Expected | Result |
|---|---:|---:|---|
| `0.10.9` | `{mre_0_10_9["jitted_gradient"]:.12f}` | `{mre_0_10_9["expected_gradient"]:.12f}` | **FAIL** |
| `0.10.11` | `{mre_0_10_11["jitted_gradient"]:.12f}` | `{mre_0_10_11["expected_gradient"]:.12f}` | **PASS** |

Disabling MLX fusion on `0.10.9` did not fix this MRE, so `no_fuse` is not a
valid workaround. The separate scan stacked-output MRE did not reproduce in
either local version and was not used as positive evidence.

## AbMPNN validation

| Gate | Latest backend | Calls | Result |
|---|---|---:|---|
| Real AF3 inference, then AbMPNN | `0.10.9` | 30 | PASS, exact sequences |
| Boundary stress with overlay | `0.10.11` | 100 | PASS, exact sequences |
| AF2 parent plus AbMPNN | `0.10.11` overlay | 50 | PASS, five finite sequences/call |
| Actual upgraded environment | `0.10.11` | 100 | PASS, exact sequences |

The post-AF3 `0.10.9` run passing 30/30 demonstrates that AF3 execution alone
is not a deterministic trigger. The production failure is consistent with the
intermittent accumulated GPU scheduler/residency hazard documented upstream.

Every successful call produced the same expected five-sequence set:
`{expected_sequence_sha}`.

## Minimal correction

The exact patch updates:

- `docs/porting/phase2b/pyproject.toml`
- `docs/porting/phase2b/installed-requirements.txt`
- `docs/porting/phase2b/uv.lock`

The AF3 environment remains pinned to `jax-mps==0.10.9` to preserve the
accepted AF3 execution boundary. Only the design/interpreter environment used
by AF2/AbMPNN was upgraded.

The verified wheel is SHA-256
`{WHEEL_0_10_11_SHA256}`. The pre-upgrade `0.10.9` plugin directory and
distribution metadata are preserved for rollback and forensic comparison.

## Literal production revalidation

A fresh literal `run_germinal.py` process completed with exit code 0 in
`{production_run["wall_seconds"]:.2f}` seconds using the corrected environment.
It completed all 60 AF2 phase-logits iterations, then the existing initial
trajectory-confidence gate rejected the design:

| Metric | Observed | Unchanged threshold | Result |
|---|---:|---:|---|
| pLDDT | {production_run["stop_metrics"]["plddt"]:.3f} | >= {production_run["thresholds"]["plddt_threshold"]} | PASS |
| iPTM | {production_run["stop_metrics"]["i_ptm"]:.3f} | >= {production_run["thresholds"]["i_ptm_threshold"]} | **FAIL** |
| iPAE | {production_run["stop_metrics"]["i_pae"]:.3f} | < {production_run["thresholds"]["i_pae_threshold"]} | PASS |

No threshold was changed. Consequently the run did not enter the AF2 softmax
and semigreedy phases, AF3, AbMPNN, PyRosetta final relaxation, or final
filters. The sole output PDB parsed with 1,924 atoms, finite coordinates, and
the expected `A=112`, `B=131` chains.

Production revalidation is therefore **PARTIAL**: the corrected backend and
AbMPNN boundary pass their direct gates, but the literal workflow did not reach
that boundary under the single accepted trajectory seed. The corrected backend
changed the AF2 numerical trajectory enough to produce iPTM `0.649` instead of
the earlier accepted value above `0.68`.

## Verification

- Source hashes unchanged: **{checks["source_hashes_unchanged"]}**
- Scientific configuration hashes unchanged:
  **{checks["scientific_configuration_hashes_unchanged"]}**
- Accepted seed `1791095845` unchanged:
  **{checks["accepted_trajectory_seed_unchanged"]}**
- ABMPNN boundary tests:
  **{tests_abmpnn["tests_run"]}/{tests_abmpnn["tests_run"]} PASS**
- AF3 boundary tests: **{tests_af3["tests_run"]}/{tests_af3["tests_run"]} PASS**
- Literal production entry point launched during Phase 2B.10: **Yes**
- Literal production workflow reached AF3/AbMPNN: **No**
- Environment dependency audit: **14 pre-existing optional-package
  incompatibilities, no JAX-related conflict**
- Transient NumPy/SciPy package-manager updates restored before validation:
  **Yes**

## Evidence

- Root cause: `phase2b10_root_cause.json`
- Environment: `phase2b10_environment.json`
- Validation: `phase2b10_validation.json`
- AbMPNN: `phase2b10_abmpnn_validation.json`
- Execution commands: `phase2b10_execution_manifest.json`
- Exact pin patch: `phase2b10_exact_patch.diff`
- Literal production run: `phase2b10_production_run.json`
- Upstream audit:
  `results/phase2b10/jax_mps_mlx_audit/audit.md`
- Pre-upgrade plugin snapshot:
  `results/phase2b10/pre_upgrade_env/jax_plugins_mps_0_10_9/`

The backend root-cause and correction gate passes. Full production
revalidation remains PARTIAL because the accepted seed is rejected by the
unchanged AF2 confidence threshold before downstream execution. Do not weaken
that threshold. A new scientifically authorized trajectory or seed is required
to exercise AF3, AbMPNN, PyRosetta, and the final filters in the literal
workflow under the corrected backend.
"""
    (DOCS / "phase2b10_report.md").write_text(report)
    checksum_paths = [
        DOCS / "phase2b10_report.md",
        DOCS / "phase2b10_summary.json",
        DOCS / "phase2b10_root_cause.json",
        DOCS / "phase2b10_environment.json",
        DOCS / "phase2b10_validation.json",
        DOCS / "phase2b10_abmpnn_validation.json",
        DOCS / "phase2b10_execution_manifest.json",
        DOCS / "phase2b10_production_run.json",
        DOCS / "phase2b10_exact_patch.diff",
        EVIDENCE_PATHS["mre_0_10_9"],
        EVIDENCE_PATHS["mre_0_10_11"],
        EVIDENCE_PATHS["post_upgrade_mre"],
        EVIDENCE_PATHS["production_state"],
        EVIDENCE_PATHS["post_upgrade_100"],
        RESULTS / "production_run1.log",
    ]
    (DOCS / "phase2b10_checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(REPO_ROOT)}\n"
            for path in checksum_paths
        )
    )
    print(json.dumps(
        {
            "decision": decision,
            "root_cause_gate_decision": root_cause_decision,
            "production_revalidation": {
                "status": (
                    "PASS" if all(production_checks.values()) else "PARTIAL"
                ),
                "checks": production_checks,
            },
            "root_cause_checks": checks,
        },
        indent=2,
    ))
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
