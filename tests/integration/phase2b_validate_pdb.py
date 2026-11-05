#!/usr/bin/env python3
"""Validate the structure written by the Phase 2B design probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from Bio import PDB


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--target-residues", type=int, required=True)
  parser.add_argument("--binder-residues", type=int, required=True)
  args = parser.parse_args()

  structure = PDB.PDBParser(QUIET=True).get_structure(
      "phase2b", args.pdb
  )
  chains = {chain.id: len(chain) for chain in structure[0]}
  coordinates = np.asarray(
      [atom.coord for atom in structure.get_atoms()], dtype=np.float64
  )
  checks = {
      "two_chains": sorted(chains) == ["A", "B"],
      "target_length": chains.get("A") == args.target_residues,
      "binder_length": chains.get("B") == args.binder_residues,
      "finite_coordinates": bool(np.isfinite(coordinates).all()),
  }
  report = {
      "status": "PASS" if all(checks.values()) else "FAIL",
      "pdb": str(args.pdb.resolve()),
      "chains": chains,
      "atoms": int(coordinates.shape[0]),
      "checks": checks,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
