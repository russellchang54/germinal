#!/usr/bin/env python3
"""Validate seeded AbMPNN output through Germinal's spawned-worker path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax  # noqa: E402

if os.environ.get("JAX_PLATFORMS") == "mps":
  configure_jax("mps")


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  args = parser.parse_args()
  import jax

  from germinal.filters.redesign import run_abmpnn_redesign_pipeline

  config = yaml.safe_load(args.config.read_text())
  run_settings = config["run_settings"]
  started = time.perf_counter()
  sequences, success = run_abmpnn_redesign_pipeline(
      trajectory_pdb_af=str(args.pdb.resolve()),
      run_settings=run_settings,
      atom_distance_cutoff=run_settings["atom_distance_cutoff"],
      seed=args.seed,
  )
  report = {
      "status": "PASS" if success and len(sequences) == 5 else "FAIL",
      "seed": args.seed,
      "input_pdb": str(args.pdb.resolve()),
      "input_pdb_sha256": sha256(args.pdb.resolve()),
      "config": str(args.config.resolve()),
      "config_sha256": sha256(args.config.resolve()),
      "jax_backend": jax.default_backend(),
      "jax_devices": [str(device) for device in jax.devices()],
      "success": success,
      "sequence_count": len(sequences),
      "sequences": sequences,
      "runtime_seconds": time.perf_counter() - started,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
