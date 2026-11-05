#!/usr/bin/env python3
"""Preserve mutable production boundary artifacts before later stages replace them."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_once(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return {
        "source": str(source.relative_to(REPO_ROOT)),
        "destination": str(destination.relative_to(REPO_ROOT)),
        "source_sha256": sha256(source),
        "preserved_sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "captured_unix_seconds": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=36000)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structures = (
        run_root / "results/germinal_run/trajectories/structures"
    )
    run_summary = (
        run_root / "results/germinal_run/run_summary.txt"
    )
    started = time.monotonic()
    captured = set()
    records = []

    def capture(key: str, source: Path, destination: Path) -> None:
        if key in captured or not source.is_file():
            return
        record = copy_once(source, destination)
        record["key"] = key
        records.append(record)
        captured.add(key)
        print(json.dumps(record), flush=True)

    while time.monotonic() - started < args.timeout_seconds:
        design = "il3_nb_s1791095845"
        capture(
            "af2_trajectory_pdb",
            structures / f"{design}.pdb",
            output_dir / "af2_trajectory.pdb",
        )
        capture(
            "af3_input_json",
            structures / f"af3_inputs/{design}.json",
            output_dir / "af3_input.json",
        )
        capture(
            "af3_target_msa",
            structures / "af3_inputs/msas/target_A.a3m",
            output_dir / "af3_target_A.a3m",
        )
        capture(
            "initial_af3_pdb",
            structures / f"{design}_af3.pdb",
            output_dir / "initial_af3.pdb",
        )
        capture(
            "initial_relaxed_pdb",
            structures / f"{design}_relaxed.pdb",
            output_dir / "initial_relaxed.pdb",
        )
        capture(
            "af3_subprocess_log",
            structures / f"af3_inputs/{design}_af3_subprocess.log",
            output_dir / "af3_subprocess.log",
        )
        worker_settings = (
            structures / f"abmpnn_workers/{design}/run_settings.json"
        )
        if worker_settings.is_file():
            capture(
                "abmpnn_input_pdb",
                structures / f"{design}.pdb",
                output_dir / "abmpnn_input.pdb",
            )
            capture(
                "abmpnn_run_settings",
                worker_settings,
                output_dir / "abmpnn_run_settings.json",
            )
        if run_summary.is_file():
            break
        time.sleep(0.1)

    summary = {
        "status": "PASS",
        "run_root": str(run_root.relative_to(REPO_ROOT)),
        "run_finished": run_summary.is_file(),
        "captured_count": len(records),
        "captured": records,
        "timed_out": time.monotonic() - started >= args.timeout_seconds,
    }
    (output_dir / "boundary_snapshot_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
