#!/usr/bin/env python3
"""Validate native PyRosetta and Germinal's PyRosetta-dependent filters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _version(name: str) -> str | None:
  try:
    return metadata.version(name)
  except metadata.PackageNotFoundError:
    return None


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--dssp",
      type=Path,
      default=REPO_ROOT / ".envs/germinal-macos/bin/mkdssp",
  )
  parser.add_argument(
      "--dalphaball",
      type=Path,
      default=REPO_ROOT / ".envs/germinal-macos/bin/dalphaball",
  )
  args = parser.parse_args()

  os.environ["LIBCIFPP_DATA_DIR"] = str(
      REPO_ROOT / ".envs/germinal-macos/share/libcifpp"
  )
  started = time.perf_counter()
  import pyrosetta as pr

  pr.init(
      "-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
      f"-holes:dalphaball {args.dalphaball} "
      "-corrections::beta_nov16 true -relax:default_repeats 1"
  )
  import mdtraj

  from germinal.filters import filter_utils, pDockQ, pyrosetta_utils
  from germinal.utils import utils

  pdb = args.pdb.resolve()
  pose = pr.pose_from_pdb(str(pdb))
  scorefxn = pr.get_fa_scorefxn()
  score = float(scorefxn(pose))
  holes_type = pr.rosetta.core.scoring.ScoreType.holes
  scorefxn.set_weight(holes_type, 1.0)
  holes_total = float(scorefxn(pose))
  holes_score = float(pose.energies().total_energies()[holes_type])
  interface, interface_aa, interface_residues = (
      pyrosetta_utils.score_interface(str(pdb), "B", "A")
  )
  loop_sc, loop_sc_area = pyrosetta_utils.calculate_loop_sc(
      pose, "B", "A"
  )
  sap = pyrosetta_utils.get_sap_score(
      str(pdb),
      binder_chain="B",
      only_binder=False,
      patch_radius=7.5,
      limit_sasa=3.0,
      avg_sasa_patch_thr=1.5,
      cdrs=None,
  )
  secondary_structure = utils.calc_ss_percentage(
      str(pdb),
      {"dssp_path": str(args.dssp)},
      chain_id="B",
      return_dict=True,
      target_chain="A",
  )
  clashes = utils.calculate_clash_score(
      str(pdb), threshold=2.4, only_ca=True
  )

  module_imports = {}
  for module_name in [
      "germinal.filters.pyrosetta_utils",
      "germinal.filters.filter_utils",
      "germinal.filters.redesign",
      "germinal.filters.af3",
      "germinal.filters.chai",
      "germinal.filters.protenix",
      "germinal.filters.pDockQ",
  ]:
    try:
      __import__(module_name)
      module_imports[module_name] = {"status": "PASS", "error": None}
    except Exception as exc:  # pylint: disable=broad-exception-caught
      module_imports[module_name] = {
          "status": "FAIL",
          "error": f"{type(exc).__name__}: {exc}",
      }

  with tempfile.TemporaryDirectory(prefix="phase2b5-pyrosetta-") as temp_dir:
    relaxed_path = Path(temp_dir) / "relaxed.pdb"
    relax_started = time.perf_counter()
    pyrosetta_utils.pr_relax(str(pdb), str(relaxed_path))
    relax_runtime = time.perf_counter() - relax_started
    relaxed_trajectory = mdtraj.load(str(relaxed_path))
    relax_result = {
        "status": "PASS" if relaxed_path.is_file() else "FAIL",
        "runtime_seconds": relax_runtime,
        "size_bytes": relaxed_path.stat().st_size if relaxed_path.is_file() else 0,
        "sha256": _sha256(relaxed_path) if relaxed_path.is_file() else None,
        "atoms": int(relaxed_trajectory.n_atoms),
        "finite_coordinates": bool(
            np.isfinite(relaxed_trajectory.xyz).all()
        ),
    }

  package_path = Path(pr.__file__).resolve()
  native_library = package_path.with_name("rosetta.so")
  license_path = package_path.with_name("LICENSE.PyRosetta.md")
  report = {
      "phase": "2B.5",
      "status": (
          "PASS"
          if all(
              item["status"] == "PASS"
              for item in module_imports.values()
          )
          and relax_result["status"] == "PASS"
          else "FAIL"
      ),
      "environment": {
          "executable": sys.executable,
          "prefix": sys.prefix,
          "python": platform.python_version(),
          "machine": platform.machine(),
          "bits": struct.calcsize("P") * 8,
          "platform": platform.platform(),
      },
      "versions": {
          name: _version(name)
          for name in [
              "pyrosetta",
              "pyrosetta-installer",
              "jax",
              "jaxlib",
              "jax-mps",
              "numpy",
              "scipy",
              "torch",
              "mdtraj",
              "colabfold",
              "chai-lab",
              "torch-geometric",
              "ipSAE",
              "gemmi",
              "numba",
              "llvmlite",
          ]
      },
      "pyrosetta": {
          "package_path": str(package_path),
          "native_library": str(native_library),
          "native_library_sha256": _sha256(native_library),
          "file_output": subprocess.run(
              ["file", str(native_library)],
              check=True,
              capture_output=True,
              text=True,
          ).stdout.strip(),
          "otool_output": subprocess.run(
              ["otool", "-L", str(native_library)],
              check=True,
              capture_output=True,
              text=True,
          ).stdout,
          "license_path": str(license_path),
          "license_sha256": _sha256(license_path),
          "license_requires_valid_concurrent_non_commercial_rosetta_license": True,
          "initialization_options": pr.get_init_options(),
      },
      "pose": {
          "path": str(pdb),
          "sha256": _sha256(pdb),
          "residues": pose.total_residue(),
          "chains": pose.num_chains(),
          "finite_coordinates": bool(
              all(
                  np.isfinite(
                      [pose.residue(i).xyz(a)[k] for k in range(3)]
                  ).all()
                  for i in range(1, pose.total_residue() + 1)
                  for a in range(1, pose.residue(i).natoms() + 1)
              )
          ),
      },
      "scoring": {
          "fa_score": score,
          "holes_score": holes_score,
          "fa_score_with_holes": holes_total,
          "interface": interface,
          "interface_amino_acids": interface_aa,
          "interface_residue_count": len(interface_residues),
          "loop_shape_complementarity": float(loop_sc),
          "loop_shape_complementarity_area": float(loop_sc_area),
          "sap_total": float(sap[0]),
          "sap_cdr": float(sap[1]),
          "exposed_hydrophobic_count": len(sap[2]),
          "hydrophobic_patch_count": len(sap[3]),
          "secondary_structure": secondary_structure,
          "ca_clashes": float(clashes),
      },
      "fast_relax": relax_result,
      "module_imports": module_imports,
      "runtime_seconds": time.perf_counter() - started,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  import numpy as np

  raise SystemExit(main())
