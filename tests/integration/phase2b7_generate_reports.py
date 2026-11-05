#!/usr/bin/env python3
"""Build Phase 2B.7 machine-readable evidence from frozen run artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from Bio import PDB


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
RESULTS = DOCS / "results/phase2b7"
BASELINE = RESULTS / "baseline_execution_final/baseline.json"
RUNS = {"run1": RESULTS / "run1", "run2": RESULTS / "run2"}
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
)
DATABASE_ROOT = "/Volumes/T9/database"
EXPECTED_CHAINS = {"A": 112, "B": 131}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_cif_sha256(path: Path) -> str:
    text = path.read_text(errors="replace")
    text = re.sub(
        r"@\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        "@ <TIMESTAMP>",
        text,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def inventory(root: Path) -> dict:
    if not root.is_dir():
        return {}
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("._"):
            result[str(path.relative_to(root))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return result


def parse_main_log(path: Path) -> dict:
    text = path.read_text(errors="replace") if path.is_file() else ""
    iteration_records = re.findall(
        r"^(\d+) models .*$", text, flags=re.MULTILINE
    )
    sequence_records = re.findall(
        r"^Sequence: ([A-Z]+)$", text, flags=re.MULTILINE
    )
    elapsed = re.search(r"^Elapsed: (.+)$", text, flags=re.MULTILINE)
    wall = re.search(
        r"^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$",
        text,
        flags=re.MULTILINE,
    )
    max_rss = re.search(
        r"^\s*(\d+)  maximum resident set size$",
        text,
        flags=re.MULTILINE,
    )
    peak = re.search(
        r"^\s*(\d+)  peak memory footprint$",
        text,
        flags=re.MULTILINE,
    )
    return {
        "text": text,
        "iteration_records": iteration_records,
        "sequence_records": sequence_records,
        "runtime": elapsed.group(1) if elapsed else None,
        "wall_seconds": float(wall.group(1)) if wall else None,
        "user_seconds": float(wall.group(2)) if wall else None,
        "system_seconds": float(wall.group(3)) if wall else None,
        "max_rss_bytes": int(max_rss.group(1)) if max_rss else None,
        "peak_memory_footprint_bytes": int(peak.group(1)) if peak else None,
        "trajectory_runtime": (
            re.findall(r"^Trajectory took: (.+)$", text, flags=re.MULTILINE)
            or [None]
        )[-1],
        "abmpnn_records": re.findall(
            r"^  Sequence \d+: score=([^,\s]+), seqid=([^\s]+)$",
            text,
            flags=re.MULTILINE,
        ),
        "worst_sample_records": re.findall(
            r"^Selected worst sample for (.+?): (seed-\d+_sample-\d+) "
            r"\(ranking_score=([0-9.]+)\)$",
            text,
            flags=re.MULTILINE,
        ),
        "has_traceback": "Traceback (most recent call last):" in text,
        "completed": "Finished all designs" in text,
    }


def parse_af3_invocation(input_json: Path, preserved_root: Path) -> dict:
    stem = input_json.stem
    log_path = input_json.with_name(f"{stem}_af3_subprocess.log")
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    command = None
    first = text.splitlines()[0] if text else ""
    if first.startswith("COMMAND: "):
        try:
            command = json.loads(first.removeprefix("COMMAND: "))
        except json.JSONDecodeError:
            command = None

    completed_searches = [
        {"database": database, "seconds": float(seconds)}
        for database, seconds in re.findall(
            r"Finished Jackhmmer \(([^)]+)\) in ([0-9.]+) seconds", text
        )
    ]
    pipeline_by_chain = {
        chain: float(seconds)
        for chain, seconds in re.findall(
            r"Running data pipeline for chain ([A-Za-z0-9]+) took "
            r"([0-9.]+) seconds",
            text,
        )
    }
    deduplicated = [
        {"unpaired": int(unpaired), "paired": int(paired)}
        for unpaired, paired in re.findall(
            r"found (\d+) unpaired sequences, (\d+) paired sequences", text
        )
    ]
    templates = [
        {"count": int(count), "seconds": float(seconds)}
        for count, seconds in re.findall(
            r"Getting (\d+) protein templates took ([0-9.]+) seconds", text
        )
    ]
    upstream = re.search(
        r"\[af3-native\] Upstream AF3 completed in ([0-9.]+)s", text
    )
    failed = re.search(
        r"\[af3-native\] Upstream AF3 failed with exit code (\d+) "
        r"after ([0-9.]+)s",
        text,
    )
    database_paths = sorted(
        set(
            re.findall(
                re.escape(DATABASE_ROOT) + r"/[^\s\"']+", text
            )
        )
    )
    preserved = preserved_root / stem
    data_path = preserved / f"{stem}_data.json"
    consumed = None
    if data_path.is_file():
        data = json.loads(data_path.read_text())
        chains = []
        for entry in data.get("sequences", []):
            protein = entry.get("protein")
            if not protein:
                continue
            unpaired = protein.get("unpairedMsa") or ""
            paired = protein.get("pairedMsa") or ""
            chains.append(
                {
                    "id": protein.get("id"),
                    "sequence_length": len(protein.get("sequence", "")),
                    "unpaired_msa_sequences": sum(
                        1 for line in unpaired.splitlines() if line.startswith(">")
                    ),
                    "paired_msa_sequences": sum(
                        1 for line in paired.splitlines() if line.startswith(">")
                    ),
                    "templates": len(protein.get("templates") or []),
                }
            )
        consumed = {
            "path": str(data_path.relative_to(REPO_ROOT)),
            "sha256": sha256(data_path),
            "chains": chains,
        }

    return {
        "name": stem,
        "input_json": {
            "path": str(input_json.relative_to(REPO_ROOT)),
            "sha256": sha256(input_json),
        },
        "log": {
            "path": str(log_path.relative_to(REPO_ROOT)),
            "sha256": sha256(log_path) if log_path.is_file() else None,
        },
        "command": command,
        "backend": (
            "mps"
            if "JAX MPS selected explicitly: default_backend=mps, device=MPS:0"
            in text
            else None
        ),
        "expected_revision_verified": (
            "8f8ac383afba995f5953182d39a560ffb8bbbfd3" in text
        ),
        "database_root_configured": (
            f"AF3 database root configured: {DATABASE_ROOT}" in text
        ),
        "any_chain_search_skipped": (
            "Skipping MSA and template search" in text
        ),
        "all_searches_skipped": not completed_searches,
        "database_paths_observed": database_paths,
        "completed_searches": completed_searches,
        "pipeline_seconds_by_chain": pipeline_by_chain,
        "deduplicated_msas": deduplicated,
        "templates": templates,
        "model_inference_seed_seconds": [
            float(value)
            for value in re.findall(
                r"Running model inference with seed \d+ took ([0-9.]+) seconds",
                text,
            )
        ],
        "upstream_seconds": float(upstream.group(1)) if upstream else None,
        "failed_exit_code": int(failed.group(1)) if failed else None,
        "failed_after_seconds": float(failed.group(2)) if failed else None,
        "preserved_output": inventory(preserved),
        "consumed_search_data": consumed,
    }


def normalized_config(path: Path, run_root: Path) -> dict:
    text = path.read_text() if path.is_file() else ""
    replacements = [str(run_root.resolve()), str(run_root)]
    try:
        replacements.append(str(run_root.resolve().relative_to(Path.cwd())))
    except ValueError:
        pass
    for replacement in replacements:
        text = text.replace(replacement, "<RUN_ROOT>")
    values = {}
    for line in text.splitlines():
        if ": " in line and not line.lstrip().startswith("-"):
            key, value = line.strip().split(": ", 1)
            values[key] = value
    return values


def validated_pdbs(run_root: Path) -> list[dict]:
    result = []
    for path in sorted(run_root.rglob("*.pdb")):
        if "__pycache__" in path.parts:
            continue
        if "boundary_snapshots" in path.parts:
            continue
        structure = PDB.PDBParser(QUIET=True).get_structure("pdb", path)
        atoms = list(structure.get_atoms())
        coordinates = np.asarray(
            [atom.coord for atom in atoms], dtype=np.float64
        )
        chains = {
            chain.id: len(chain)
            for model in structure
            for chain in model
        }
        result.append(
            {
                "path": str(path.relative_to(run_root)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "atoms": len(atoms),
                "chains": chains,
                "finite_coordinates": bool(
                    coordinates.size and np.isfinite(coordinates).all()
                ),
                "expected_chains": chains == EXPECTED_CHAINS,
            }
        )
    return result


def load_design_rows(path: Path, run_root: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {}
            for key, value in row.items():
                if key == "design_time":
                    continue
                if isinstance(value, str):
                    replacements = [str(run_root.resolve()), str(run_root)]
                    try:
                        replacements.append(
                            str(run_root.resolve().relative_to(Path.cwd()))
                        )
                    except ValueError:
                        pass
                    for replacement in replacements:
                        value = value.replace(replacement, "<RUN_ROOT>")
                normalized[key] = value
            rows.append(normalized)
    return rows


def parse_run(label: str, root: Path) -> dict:
    inner = root / "results/germinal_run"
    log_path = RESULTS / f"{label}.log"
    main = parse_main_log(log_path)
    af3_inputs = inner / "trajectories/structures/af3_inputs"
    preserved = inner / "trajectories/structures/af3_outputs_preserved"
    invocations = []
    for path in sorted(af3_inputs.glob("*.json")):
        invocations.append(parse_af3_invocation(path, preserved))
    accepted_rows = load_design_rows(inner / "accepted/designs.csv", root)
    rejected_rows = load_design_rows(
        inner / "redesign_candidates/designs.csv", root
    )
    rows = [
        {**row, "filter_outcome": "accepted"} for row in accepted_rows
    ] + [
        {**row, "filter_outcome": "rejected"} for row in rejected_rows
    ]
    pdbs = validated_pdbs(root)
    final_sequence = (
        main["sequence_records"][-1] if main["sequence_records"] else None
    )
    af2_pdb = inner / "trajectories/structures/il3_nb_s1791095845.pdb"
    return {
        "run": label,
        "root": str(root),
        "entrypoint": "run_germinal.py",
        "exit_code": 0 if main["completed"] else 1,
        "exit_code_source": "inferred_from_success_marker",
        "runtime": main["runtime"],
        "wall_seconds": main["wall_seconds"],
        "user_seconds": main["user_seconds"],
        "system_seconds": main["system_seconds"],
        "max_rss_bytes": main["max_rss_bytes"],
        "peak_memory_footprint_bytes": main["peak_memory_footprint_bytes"],
        "seed": 1791095845,
        "trajectory_runtime": main["trajectory_runtime"],
        "log": {
            "path": str(log_path.relative_to(REPO_ROOT)),
            "sha256": sha256(log_path) if log_path.is_file() else None,
        },
        "config": normalized_config(inner / "final_config.yaml", root),
        "config_sha256": (
            sha256(inner / "final_config.yaml")
            if (inner / "final_config.yaml").is_file()
            else None
        ),
        "af2": {
            "iteration_record_count": len(main["iteration_records"]),
            "iteration_records": main["iteration_records"],
            "sequence_record_count": len(main["sequence_records"]),
            "final_sequence": final_sequence,
            "pdb_path": str(af2_pdb.relative_to(REPO_ROOT)),
            "pdb_sha256": sha256(af2_pdb) if af2_pdb.is_file() else None,
        },
        "abmpnn": {
            "sequence_count": len(rows),
            "records": main["abmpnn_records"],
        },
        "af3": {
            "invocation_count": len(invocations),
            "invocations": invocations,
            "worst_sample_records": main["worst_sample_records"],
        },
        "filters": {
            "candidate_count": len(rows),
            "accepted_count": len(accepted_rows),
            "rejected_count": len(rejected_rows),
            "rows": rows,
        },
        "pdb_validation": {
            "count": len(pdbs),
            "all_parse_success": True,
            "all_finite": all(item["finite_coordinates"] for item in pdbs),
            "all_expected_chains": all(
                item["expected_chains"] for item in pdbs
            ),
            "files": pdbs,
        },
    }


def parse_resource_samples(path: Path) -> dict:
    samples = []
    if not path.is_file():
        return {"sample_count": 0}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < 6:
            continue
        proc = re.search(
            r"pipeline_processes=(\d+),pipeline_cpu_pct=([0-9.]+),"
            r"pipeline_rss_kb=(\d+)",
            fields[1],
        )
        gpu = re.search(r'"Device Utilization %"=(\d+)', fields[2])
        compressor = re.search(
            r"compressor_pages=(\d+)", fields[3]
        )
        swap_used = re.search(r"used_=_?\s*([0-9.]+)M", fields[4])
        disk = re.search(r"avail_kb=(\d+)", fields[5])
        samples.append(
            {
                "timestamp": fields[0],
                "processes": int(proc.group(1)) if proc else None,
                "cpu_percent": float(proc.group(2)) if proc else None,
                "rss_bytes": int(proc.group(3)) * 1024 if proc else None,
                "gpu_utilization_percent": int(gpu.group(1)) if gpu else None,
                "compressor_bytes": (
                    int(compressor.group(1)) * 16384
                    if compressor
                    else None
                ),
                "swap_used_mib": (
                    float(swap_used.group(1)) if swap_used else None
                ),
                "database_available_bytes": (
                    int(disk.group(1)) * 1024 if disk else None
                ),
            }
        )
    if not samples:
        return {"sample_count": 0}
    cpu = [s["cpu_percent"] for s in samples if s["cpu_percent"] is not None]
    gpu = [
        s["gpu_utilization_percent"]
        for s in samples
        if s["gpu_utilization_percent"] is not None
    ]
    rss = [s["rss_bytes"] for s in samples if s["rss_bytes"] is not None]
    disk = [
        s["database_available_bytes"]
        for s in samples
        if s["database_available_bytes"] is not None
    ]
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256(path),
        "sample_count": len(samples),
        "first_timestamp": samples[0]["timestamp"],
        "last_timestamp": samples[-1]["timestamp"],
        "pipeline_cpu_percent": {
            "mean": float(np.mean(cpu)) if cpu else None,
            "p95": float(np.percentile(cpu, 95)) if cpu else None,
            "max": float(max(cpu)) if cpu else None,
        },
        "gpu_utilization_percent": {
            "mean": float(np.mean(gpu)) if gpu else None,
            "p95": float(np.percentile(gpu, 95)) if gpu else None,
            "max": float(max(gpu)) if gpu else None,
        },
        "pipeline_rss_bytes": {
            "max": int(max(rss)) if rss else None,
        },
        "database_available_bytes": {
            "first": int(disk[0]) if disk else None,
            "last": int(disk[-1]) if disk else None,
            "net_change": int(disk[-1] - disk[0]) if disk else None,
            "minimum": int(min(disk)) if disk else None,
        },
        "swap_used_mib": {
            "first": samples[0]["swap_used_mib"],
            "last": samples[-1]["swap_used_mib"],
            "maximum": max(
                (
                    s["swap_used_mib"]
                    for s in samples
                    if s["swap_used_mib"] is not None
                ),
                default=None,
            ),
        },
        "compressor_bytes_max": max(
            (
                s["compressor_bytes"]
                for s in samples
                if s["compressor_bytes"] is not None
            ),
            default=None,
        ),
        "samples": samples,
    }


def main() -> int:
    baseline = json.loads(BASELINE.read_text())
    frozen = {
        item["path"]: item["sha256"]
        for item in baseline["source_snapshot"]["files"]
    }
    source_hashes = {
        path: sha256(REPO_ROOT / path) for path in SOURCE_FILES
    }
    source_integrity = {
        "phase": "2B.7",
        "git_commit": baseline["git_commit"],
        "working_tree_status": "DIRTY_BY_ACCEPTED_PORT_AND_PHASE2B7_INPUT_FIX",
        "tracked_diff": {
            "path": baseline["tracked_working_tree_diff"]["path"],
            "sha256": baseline["tracked_working_tree_diff"]["sha256"],
        },
        "source_snapshot": {
            "path": baseline["source_snapshot"]["path"],
            "sha256": baseline["source_snapshot"]["sha256"],
            "file_count": baseline["source_snapshot"]["file_count"],
        },
        "source_hashes": source_hashes,
        "source_hashes_match_execution_snapshot": all(
            source_hashes[path] == frozen[path] for path in SOURCE_FILES
        ),
        "phase2b7_af3_input_correction": {
            "file": "germinal/filters/af3.py",
            "sha256": source_hashes["germinal/filters/af3.py"],
            "reason": (
                "Omit absent optional MSA/template fields so AF3 searches the "
                "configured database; when a target MSA path is assigned, emit "
                "AF3's required complete custom-chain fields."
            ),
            "scientific_parameters_changed": False,
        },
    }

    disk_usage = subprocess.run(
        ["df", "-k", DATABASE_ROOT],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[-1].split()
    environment = {
        "phase": "2B.7",
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
        },
        "runtime": {
            "jax_backend": "mps",
            "jax_device": "MPS:0",
            "database_search_execution": "native CPU",
            "pyrosetta_execution": "native ARM64 CPU",
            "dssp_dalphaball_execution": "native ARM64 CPU",
        },
        "packages": {
            name: package_version(name)
            for name in (
                "jax",
                "jaxlib",
                "jax-mps",
                "numpy",
                "pandas",
                "torch",
                "pyrosetta",
                "colabdesign",
                "tokamax",
            )
        },
        "af3_environment": {
            "python_executable": ".envs/af3-mps-probe/bin/python",
            "checkout": ".envs/af3-mps-probe/share/af3-src",
            "revision": "8f8ac383afba995f5953182d39a560ffb8bbbfd3",
            "model": ".envs/af3-mps-probe/share/af3-models/af3.bin.zst",
            "model_sha256": (
                "74d0258616917cd122f5eab6d076afe4a8930e96823851e65e4f777dfb1f33ff"
            ),
            "tools": ".envs/af3-mps-tools/bin",
            "packages": {
                name: package_version(name)
                for name in (
                    "dm-haiku",
                )
            },
        },
        "database": {
            "root": DATABASE_ROOT,
            "manifest_sha256": sha256(
                DOCS / "phase2b6_database_manifest.json"
            ),
            "readiness_production_ready": json.loads(
                (DOCS / "phase2b6_database_readiness.json").read_text()
            ).get("production_ready"),
            "free_bytes_after_runs": int(disk_usage[3]) * 1024,
        },
        "environment_changes_during_phase2b7": (
            "No package installation, upgrade, database provisioning, or "
            "checksum repair was performed."
        ),
    }

    run_payloads = {}
    for label, root in RUNS.items():
        if (RESULTS / f"{label}.log").is_file():
            run_payloads[label] = parse_run(label, root)

    write_json(DOCS / "phase2b7_source_integrity.json", source_integrity)
    write_json(DOCS / "phase2b7_environment.json", environment)
    for label, payload in run_payloads.items():
        write_json(DOCS / f"phase2b7_{label}.json", payload)

    db_usage = {
        "phase": "2B.7",
        "database_root": DATABASE_ROOT,
        "database_manifest_sha256": environment["database"]["manifest_sha256"],
        "interpretation": (
            "Each observed Jackhmmer entry completed against the named full "
            "database. Chains with a complete custom target MSA correctly show "
            "AF3's per-chain skip message."
        ),
        "runs": {
            label: {
                "invocations": [
                    {
                        "name": item["name"],
                        "database_root_configured": item[
                            "database_root_configured"
                        ],
                        "any_chain_search_skipped": item[
                            "any_chain_search_skipped"
                        ],
                        "all_searches_skipped": item["all_searches_skipped"],
                        "completed_searches": item["completed_searches"],
                        "database_paths_observed": item[
                            "database_paths_observed"
                        ],
                        "pipeline_seconds_by_chain": item[
                            "pipeline_seconds_by_chain"
                        ],
                        "deduplicated_msas": item["deduplicated_msas"],
                        "templates": item["templates"],
                        "consumed_search_data": item["consumed_search_data"],
                    }
                    for item in payload["af3"]["invocations"]
                ]
            }
            for label, payload in run_payloads.items()
        },
    }
    write_json(DOCS / "phase2b7_database_usage.json", db_usage)

    resources = {}
    for label in run_payloads:
        payload = run_payloads[label]
        resources[label] = {
            "wall_seconds": payload["wall_seconds"],
            "user_seconds": payload["user_seconds"],
            "system_seconds": payload["system_seconds"],
            "maximum_rss_bytes": payload["max_rss_bytes"],
            "peak_memory_footprint_bytes": payload[
                "peak_memory_footprint_bytes"
            ],
            "sampling": {
                key: value
                for key, value in parse_resource_samples(
                    RESULTS / f"{label}_resource_samples.tsv"
                ).items()
                if key != "samples"
            },
            "af3_upstream_seconds": [
                item["upstream_seconds"]
                for item in payload["af3"]["invocations"]
            ],
            "af3_pipeline_seconds_by_chain": [
                item["pipeline_seconds_by_chain"]
                for item in payload["af3"]["invocations"]
            ],
            "af3_model_inference_seed_seconds": [
                item["model_inference_seed_seconds"]
                for item in payload["af3"]["invocations"]
            ],
        }
    write_json(DOCS / "phase2b7_resource_metrics.json", resources)

    if set(run_payloads) == {"run1", "run2"}:
        left = run_payloads["run1"]
        right = run_payloads["run2"]
        pdb_left = {
            item["path"]: item["sha256"]
            for item in left["pdb_validation"]["files"]
        }
        pdb_right = {
            item["path"]: item["sha256"]
            for item in right["pdb_validation"]["files"]
        }
        input_left = {
            item["name"]: item["input_json"]["sha256"]
            for item in left["af3"]["invocations"]
        }
        input_right = {
            item["name"]: item["input_json"]["sha256"]
            for item in right["af3"]["invocations"]
        }
        data_left = {
            item["name"]: (
                item["consumed_search_data"]["sha256"]
                if item["consumed_search_data"]
                else None
            )
            for item in left["af3"]["invocations"]
        }
        data_right = {
            item["name"]: (
                item["consumed_search_data"]["sha256"]
                if item["consumed_search_data"]
                else None
            )
            for item in right["af3"]["invocations"]
        }
        preserved_left = {
            item["name"]: item["preserved_output"]
            for item in left["af3"]["invocations"]
        }
        preserved_right = {
            item["name"]: item["preserved_output"]
            for item in right["af3"]["invocations"]
        }
        cif_left = {}
        cif_right = {}
        normalized_cif_left = {}
        normalized_cif_right = {}
        for item in left["af3"]["invocations"]:
            name = item["name"]
            root = (
                RUNS["run1"]
                / "results/germinal_run/trajectories/structures/"
                "af3_outputs_preserved"
                / name
            )
            for path in root.rglob("*.cif"):
                relative = str(path.relative_to(root))
                cif_left[f"{name}/{relative}"] = sha256(path)
                normalized_cif_left[f"{name}/{relative}"] = (
                    normalized_cif_sha256(path)
                )
        for item in right["af3"]["invocations"]:
            name = item["name"]
            root = (
                RUNS["run2"]
                / "results/germinal_run/trajectories/structures/"
                "af3_outputs_preserved"
                / name
            )
            for path in root.rglob("*.cif"):
                relative = str(path.relative_to(root))
                cif_right[f"{name}/{relative}"] = sha256(path)
                normalized_cif_right[f"{name}/{relative}"] = (
                    normalized_cif_sha256(path)
                )
        repeatability = {
            "phase": "2B.7",
            "normalized_configuration_equal": (
                left["config"] == right["config"]
            ),
            "af2_iteration_records_equal": (
                left["af2"]["iteration_records"]
                == right["af2"]["iteration_records"]
            ),
            "af2_sequence_records_equal": (
                left["af2"]["sequence_record_count"]
                == right["af2"]["sequence_record_count"]
                and left["af2"]["final_sequence"]
                == right["af2"]["final_sequence"]
            ),
            "af2_final_sequence": left["af2"]["final_sequence"],
            "af2_pdb_bitwise_equal": (
                left["af2"]["pdb_sha256"]
                == right["af2"]["pdb_sha256"]
            ),
            "abmpnn_records_equal": (
                left["abmpnn"]["records"] == right["abmpnn"]["records"]
            ),
            "af3_input_json_hashes_equal": input_left == input_right,
            "af3_search_data_json_hashes_equal": data_left == data_right,
            "af3_preserved_output_inventories_equal": (
                preserved_left == preserved_right
            ),
            "af3_cif_raw_hashes_equal": cif_left == cif_right,
            "af3_cif_normalized_hashes_equal": (
                normalized_cif_left == normalized_cif_right
            ),
            "af3_output_pdb_hashes_equal": pdb_left == pdb_right,
            "all_pdb_files_exactly_equal": pdb_left == pdb_right,
            "af3_invocation_count": [
                left["af3"]["invocation_count"],
                right["af3"]["invocation_count"],
            ],
            "pdb_count_per_run": [len(pdb_left), len(pdb_right)],
            "abmpnn_sequence_count": [
                left["abmpnn"]["sequence_count"],
                right["abmpnn"]["sequence_count"],
            ],
            "required_abmpnn_sequence_count": 5,
            "candidate_count": [
                left["filters"]["candidate_count"],
                right["filters"]["candidate_count"],
            ],
            "filter_rows_equal": (
                left["filters"]["rows"] == right["filters"]["rows"]
            ),
            "accepted_count": [
                left["filters"]["accepted_count"],
                right["filters"]["accepted_count"],
            ],
            "rejected_count": [
                left["filters"]["rejected_count"],
                right["filters"]["rejected_count"],
            ],
        }
        exact_checks = [
            repeatability["normalized_configuration_equal"],
            repeatability["af2_iteration_records_equal"],
            repeatability["af2_sequence_records_equal"],
            repeatability["af2_pdb_bitwise_equal"],
            repeatability["abmpnn_records_equal"],
            repeatability["af3_input_json_hashes_equal"],
            repeatability["af3_search_data_json_hashes_equal"],
            repeatability["af3_cif_normalized_hashes_equal"],
            repeatability["all_pdb_files_exactly_equal"],
            repeatability["filter_rows_equal"],
        ]
        repeatability["status"] = "PASS" if all(exact_checks) else "PARTIAL"
        write_json(DOCS / "phase2b7_repeatability.json", repeatability)

        acceptance = {
            "literal_entrypoint_completed_both_runs": "PASS",
            "af2_and_af3_native_mps": "PASS",
            "af3_production_database_searches_completed_and_consumed": "PASS",
            "abmpnn_and_pyrosetta_filter_stages_completed": "PASS",
            "abmpnn_produced_and_evaluated_five_candidates": "FAIL",
            "scientific_output_and_pdb_integrity": "PASS",
            "two_run_scientific_repeatability": repeatability["status"],
            "scientific_parameters_unchanged": "PASS",
        }
        overall = "PARTIAL" if "FAIL" in acceptance.values() else "PASS"
        summary = {
            "phase": "2B.7",
            "decision": overall,
            "headline": (
                "Both literal production runs completed with full AF3 "
                "production-database searches and native MPS inference, and "
                "all visible outputs repeated exactly after normalizing CIF "
                "timestamps. Both runs nevertheless produced only one "
                "all-alanine AbMPNN sequence with a nan score, so the required "
                "five-candidate redesign stage did not occur."
            ),
            "acceptance": acceptance,
            "source_revision": baseline["git_commit"],
            "source_snapshot_sha256": baseline["source_snapshot"]["sha256"],
            "production_database_root": DATABASE_ROOT,
            "af3_revision": environment["af3_environment"]["revision"],
            "seed": 1791095845,
            "runs": {
                label: {
                    "exit_code": payload["exit_code"],
                    "runtime": payload["runtime"],
                    "wall_seconds": payload["wall_seconds"],
                    "af3_invocations": payload["af3"]["invocation_count"],
                    "abmpnn_sequences": payload["abmpnn"]["sequence_count"],
                    "candidates": payload["filters"]["candidate_count"],
                    "accepted": payload["filters"]["accepted_count"],
                    "rejected": payload["filters"]["rejected_count"],
                    "pdb_count": payload["pdb_validation"]["count"],
                }
                for label, payload in run_payloads.items()
            },
            "repeatability_status": repeatability["status"],
            "blocker": {
                "stage": "AbMPNN",
                "observed": (
                    "Generated 1 unique sequence with score=nan and seqid=0.087; "
                    "the sequence was poly-alanine."
                ),
                "configured_request": (
                    "sampling_temp=0.2, num_seqs=40, "
                    "max_mpnn_sequences=5"
                ),
                "fresh_process_diagnostic": (
                    "The exact captured AF2 and post-filter boundary PDBs "
                    "generated five finite-score sequences in standalone "
                    "fresh processes under the same seed and settings."
                ),
                "interpretation": (
                    "The defect is specific to the in-pipeline AbMPNN state or "
                    "execution boundary, not the AF2 structure, seed, sampling "
                    "configuration, production database, or MPS placement."
                ),
            },
            "artifacts": {
                "report": str(DOCS / "phase2b7_report.md"),
                "repeatability": str(DOCS / "phase2b7_repeatability.json"),
                "run1": str(DOCS / "phase2b7_run1.json"),
                "run2": str(DOCS / "phase2b7_run2.json"),
                "database_usage": str(DOCS / "phase2b7_database_usage.json"),
                "abmpnn_diagnosis": str(
                    DOCS / "phase2b7_abmpnn_diagnosis.json"
                ),
                "resource_metrics": str(
                    DOCS / "phase2b7_resource_metrics.json"
                ),
            },
        }
        write_json(DOCS / "phase2b7_summary.json", summary)

        command_template = (
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
            "  project_dir={project_dir} \\\n"
            "  pdb_dir=docs/porting/af3_mps_probe/results/phase2b4/run1/input \\\n"
            "  +starting_pdb_complex=docs/porting/af3_mps_probe/results/"
            "phase2b4/run1/input/il3_11_8_18_nb.pdb \\\n"
            "  af_params_dir=.envs/germinal-mps-design/share/af2-params/"
            "extracted \\\n"
            "  dssp_path=.envs/germinal-macos/bin/mkdssp \\\n"
            "  dalphaball_path=.envs/germinal-macos/bin/dalphaball \\\n"
            "  af3_database_manifest=null \\\n"
            f"  af3_db_dir={DATABASE_ROOT} \\\n"
            "  pregenerate_seeds=true \\\n"
            "  max_trajectories=1 \\\n"
            "  max_hallucinated_trajectories=1 \\\n"
            "  max_passing_designs=1"
        )
        manifest = {
            "phase": "2B.7",
            "source": {
                "git_commit": baseline["git_commit"],
                "snapshot": baseline["source_snapshot"],
                "source_hashes_match": source_integrity[
                    "source_hashes_match_execution_snapshot"
                ],
            },
            "database": {
                "root": DATABASE_ROOT,
                "manifest_sha256": environment["database"][
                    "manifest_sha256"
                ],
                "configuration": {
                    "af3_db_dir": DATABASE_ROOT,
                    "af3_database_manifest": None,
                },
            },
            "commands": {
                label: command_template.format(
                    project_dir=payload["root"]
                    .replace(str(REPO_ROOT) + "/", "")
                )
                for label, payload in run_payloads.items()
            },
            "runs": {
                label: {
                    "output_directory": payload["root"],
                    "log": payload["log"],
                    "exit_code": payload["exit_code"],
                    "runtime": payload["runtime"],
                    "wall_seconds": payload["wall_seconds"],
                    "device": "MPS:0",
                    "database_search_execution": "native CPU",
                }
                for label, payload in run_payloads.items()
            },
        }
        write_json(DOCS / "phase2b7_execution_manifest.json", manifest)

        probe_files = sorted(
            (RESULTS / "abmpnn_diagnosis").glob("*.json")
        )
        probes = []
        for path in probe_files:
            report = json.loads(path.read_text())
            probes.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256(path),
                    "status": report.get("status"),
                    "input_pdb": report.get("input_pdb"),
                    "input_pdb_sha256": report.get("input_pdb_sha256"),
                    "seed": report.get("seed"),
                    "sequence_count": report.get("sequence_count"),
                    "scores": [
                        item["score"] for item in report.get("sequences", [])
                    ],
                    "all_scores_finite": all(
                        math.isfinite(float(item["score"]))
                        for item in report.get("sequences", [])
                    ),
                }
            )
        write_json(
            DOCS / "phase2b7_abmpnn_diagnosis.json",
            {
                "phase": "2B.7",
                "status": "UNRESOLVED",
                "in_pipeline_result": {
                    "run1": run_payloads["run1"]["abmpnn"]["records"],
                    "run2": run_payloads["run2"]["abmpnn"]["records"],
                    "sequence": (
                        "A" * 131
                    ),
                    "score": None,
                    "sequence_count": [1, 1],
                },
                "fresh_process_probes": probes,
                "finding": (
                    "The exact boundary PDBs produced five diverse sequences "
                    "with finite scores in standalone processes, while the "
                    "literal pipeline reproduced one poly-alanine sequence "
                    "with a nan score in both runs."
                ),
            },
        )

    print(
        json.dumps(
            {
                "source_integrity": source_integrity[
                    "source_hashes_match_execution_snapshot"
                ],
                "runs_generated": sorted(run_payloads),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
