#!/usr/bin/env python3
"""Isolate PyRosetta FastRelax RNG behavior on a fixed AF3 structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--output-pdb", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--seed", default="none")
  args = parser.parse_args()

  import pyrosetta as pr

  pr.init(
      "-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
      f"-holes:dalphaball {REPO_ROOT / '.envs/germinal-macos/bin/dalphaball'} "
      "-corrections::beta_nov16 true -relax:default_repeats 1"
  )
  rng = pr.rosetta.numeric.random.rg()
  requested_seed = (
      None if args.seed.lower() in {"none", "null"} else int(args.seed)
  )

  from germinal.filters.pyrosetta_utils import pr_relax

  args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
  started = time.perf_counter()
  pr_relax(
      str(args.pdb.resolve()),
      str(args.output_pdb.resolve()),
      seed=requested_seed,
  )
  report = {
      "status": "PASS" if args.output_pdb.is_file() else "FAIL",
      "python_pid": os.getpid(),
      "input_pdb": str(args.pdb.resolve()),
      "input_sha256": sha256(args.pdb.resolve()),
      "requested_seed": requested_seed,
      "rosetta_seed_after_setup": rng.get_seed(),
      "output_pdb": str(args.output_pdb.resolve()),
      "output_sha256": (
          sha256(args.output_pdb.resolve())
          if args.output_pdb.is_file()
          else None
      ),
      "runtime_seconds": time.perf_counter() - started,
  }
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
