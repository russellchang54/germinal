#!/usr/bin/env python3
"""Compare matched CPU and MPS Germinal design probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from Bio import PDB


def _array_metrics(mps: np.ndarray, cpu: np.ndarray) -> dict:
  delta = np.abs(mps.astype(np.float64) - cpu.astype(np.float64))
  denominator = np.maximum(
      np.maximum(np.abs(mps.astype(np.float64)), np.abs(cpu.astype(np.float64))),
      1e-12,
  )
  mps_flat = mps.astype(np.float64).ravel()
  cpu_flat = cpu.astype(np.float64).ravel()
  cosine = float(
      np.dot(mps_flat, cpu_flat)
      / (np.linalg.norm(mps_flat) * np.linalg.norm(cpu_flat) + 1e-12)
  )
  return {
      "shape": list(mps.shape),
      "equal": bool(np.array_equal(mps, cpu)),
      "max_abs_error": float(delta.max()),
      "mean_abs_error": float(delta.mean()),
      "rms_error": float(np.sqrt(np.mean(delta**2))),
      "max_relative_error": float((delta / denominator).max()),
      "cosine_similarity": cosine,
      "allclose_rtol_1e-4_atol_1e-4": bool(
          np.allclose(mps, cpu, rtol=1e-4, atol=1e-4)
      ),
  }


def _ca_coordinates(path: Path) -> np.ndarray:
  structure = PDB.PDBParser(QUIET=True).get_structure("model", path)
  return np.asarray(
      [
          atom.coord
          for atom in structure.get_atoms()
          if atom.get_name() == "CA"
      ],
      dtype=np.float64,
  )


def _kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
  if reference.shape != mobile.shape:
    raise ValueError(f"Coordinate shapes differ: {reference.shape} vs {mobile.shape}")
  ref_centered = reference - reference.mean(axis=0)
  mobile_centered = mobile - mobile.mean(axis=0)
  covariance = mobile_centered.T @ ref_centered
  u, _, vt = np.linalg.svd(covariance)
  rotation = vt.T @ u.T
  if np.linalg.det(rotation) < 0:
    vt[-1] *= -1
    rotation = vt.T @ u.T
  aligned = mobile_centered @ rotation.T
  return float(
      np.sqrt(np.mean(np.sum((aligned - ref_centered) ** 2, axis=1)))
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--steps", type=int, required=True)
  parser.add_argument("--target-residues", type=int, default=31)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  mps_dir = args.root / "mps"
  cpu_dir = args.root / "cpu"
  def artifact(directory: Path, prefix: str, suffix: str) -> Path:
    tagged = directory / f"{prefix}_t{args.target_residues}_{suffix}"
    if tagged.is_file():
      return tagged
    legacy = directory / f"{prefix}_{suffix}"
    if legacy.is_file():
      return legacy
    raise FileNotFoundError(
        f"Neither tagged nor legacy comparison artifact exists: {tagged}, {legacy}"
    )

  mps_report = json.loads(
      artifact(
          mps_dir, "phase2b_design", f"{args.steps}step_report.json"
      ).read_text()
  )
  cpu_report = json.loads(
      artifact(
          cpu_dir, "phase2b_design", f"{args.steps}step_report.json"
      ).read_text()
  )
  initial_mps = np.load(
      artifact(
          mps_dir, "initial_seq_params", f"{args.steps}step.npy"
      )
  )
  initial_cpu = np.load(
      artifact(
          cpu_dir, "initial_seq_params", f"{args.steps}step.npy"
      )
  )
  final_mps = np.load(
      artifact(
          mps_dir, "final_seq_params", f"{args.steps}step.npy"
      )
  )
  final_cpu = np.load(
      artifact(
          cpu_dir, "final_seq_params", f"{args.steps}step.npy"
      )
  )

  step_comparison = []
  for mps_step, cpu_step in zip(mps_report["steps"], cpu_report["steps"]):
    step_comparison.append(
        {
            "iteration": mps_step["iteration"],
            "loss": {
                "mps": mps_step["loss"],
                "cpu": cpu_step["loss"],
                "absolute_difference": abs(mps_step["loss"] - cpu_step["loss"]),
                "relative_difference": abs(
                    mps_step["loss"] - cpu_step["loss"]
                )
                / max(abs(cpu_step["loss"]), 1e-12),
            },
            "gradient_l2": {
                "mps": mps_step["gradient_l2"],
                "cpu": cpu_step["gradient_l2"],
                "absolute_difference": abs(
                    mps_step["gradient_l2"] - cpu_step["gradient_l2"]
                ),
            },
            "parameter_update_l2": {
                "mps": mps_step["parameter_update_l2"],
                "cpu": cpu_step["parameter_update_l2"],
                "absolute_difference": abs(
                    mps_step["parameter_update_l2"]
                    - cpu_step["parameter_update_l2"]
                ),
            },
            "plddt_difference": mps_step["plddt"] - cpu_step["plddt"],
            "iptm_difference": mps_step["iptm"] - cpu_step["iptm"],
        }
    )

  mps_pdb = Path(mps_report["final_pdb"]["path"])
  cpu_pdb = Path(cpu_report["final_pdb"]["path"])
  ca_rmsd = _kabsch_rmsd(_ca_coordinates(cpu_pdb), _ca_coordinates(mps_pdb))
  report = {
      "steps": args.steps,
      "execution_status": (
          "PASS"
          if mps_report["status"] == cpu_report["status"] == "PASS"
          else "FAIL"
      ),
      "devices": {
          "mps": mps_report["steps"][0]["gradient_device"],
          "cpu": cpu_report["steps"][0]["gradient_device"],
      },
      "sequence": {
          "initial_equal": (
              mps_report["initial_sequence"] == cpu_report["initial_sequence"]
          ),
          "mps_initial": mps_report["initial_sequence"],
          "cpu_initial": cpu_report["initial_sequence"],
          "final_equal": (
              mps_report["final_sequence"] == cpu_report["final_sequence"]
          ),
          "mps_final": mps_report["final_sequence"],
          "cpu_final": cpu_report["final_sequence"],
      },
      "initial_parameters": _array_metrics(initial_mps, initial_cpu),
      "final_parameters": _array_metrics(final_mps, final_cpu),
      "steps_comparison": step_comparison,
      "final_ca_kabsch_rmsd_angstrom": ca_rmsd,
      "runtime": {
          "mps_seconds": mps_report["runtime_seconds"],
          "cpu_seconds": cpu_report["runtime_seconds"],
          "speedup_cpu_over_mps": (
              cpu_report["runtime_seconds"] / mps_report["runtime_seconds"]
          ),
      },
      "memory": {
          "mps_max_rss_bytes": mps_report["max_rss_bytes"],
          "cpu_max_rss_bytes": cpu_report["max_rss_bytes"],
          "mps_reduction_bytes": (
              cpu_report["max_rss_bytes"] - mps_report["max_rss_bytes"]
          ),
      },
      "assessment": {
          "single_step_sequence_matches": None,
          "numerical_equivalence_claimed": False,
      },
  }
  if args.steps == 1:
    report["assessment"]["single_step_sequence_matches"] = report["sequence"][
        "final_equal"
    ]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0 if report["execution_status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
