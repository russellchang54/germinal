#!/usr/bin/env python3
"""Run a small genuine Germinal/ColabDesign optimization on MPS or CPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_minimal_complex(
    target_source: Path,
    binder_source: Path,
    output_path: Path,
    target_residues: int,
) -> None:
  from Bio import PDB

  parser = PDB.PDBParser(QUIET=True)
  target = parser.get_structure("target", target_source)
  binder = parser.get_structure("binder", binder_source)
  target_chain = target[0]["A"]
  binder_chain = binder[0]["A"]

  structure = PDB.Structure.Structure("phase2b_complex")
  model = PDB.Model.Model(0)
  structure.add(model)
  target_output = PDB.Chain.Chain("A")
  binder_output = PDB.Chain.Chain("B")
  model.add(target_output)
  model.add(binder_output)
  for index, residue in enumerate(target_chain):
    if index >= target_residues:
      break
    target_output.add(copy.deepcopy(residue))
  for residue in binder_chain:
    binder_output.add(copy.deepcopy(residue))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  io = PDB.PDBIO()
  io.set_structure(structure)
  io.save(str(output_path))


def _tensor_devices(value) -> list[str]:
  if isinstance(value, (list, tuple)):
    devices = set()
    for item in value:
      devices.update(_tensor_devices(item))
    return sorted(devices)
  if isinstance(value, dict):
    devices = set()
    for item in value.values():
      devices.update(_tensor_devices(item))
    return sorted(devices)
  if hasattr(value, "devices"):
    return sorted(str(device) for device in value.devices())
  return []


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", choices=["mps", "cpu"], required=True)
  parser.add_argument("--steps", type=int, default=2)
  parser.add_argument("--target-residues", type=int, default=31)
  parser.add_argument(
      "--params-dir",
      type=Path,
      default=REPO_ROOT
      / ".envs/germinal-mps-design/share/af2-params/minimal",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=REPO_ROOT
      / ".envs/germinal-mps-design/share/phase2b-design-probe",
  )
  args = parser.parse_args()

  jax = configure_jax(args.backend)
  import jax.numpy as jnp
  import numpy as np
  from hydra import compose, initialize_config_dir

  from colabdesign.af.design import _af_design
  from germinal.design.design import germinal_design
  from germinal.utils.config import process_config
  from germinal.utils.utils import compute_cdr_positions, get_sequence_from_pdb
  from germinal.utils.io import IO, RunLayout

  output_root = args.output_dir.resolve() / args.backend
  output_root.mkdir(parents=True, exist_ok=True)
  fixture_dir = REPO_ROOT / "docs/porting/af3_mps_probe/results/phase2b/input"
  fixture_path = fixture_dir / f"il3_{args.target_residues}_nb_131.pdb"
  _write_minimal_complex(
      REPO_ROOT / "pdbs/il3.pdb",
      REPO_ROOT / "pdbs/nb.pdb",
      fixture_path,
      args.target_residues,
  )

  with initialize_config_dir(
      config_dir=str(REPO_ROOT / "configs"), version_base=None
  ):
    cfg = compose(
      config_name="config",
      overrides=[
        "run=vhh_il3",
        "target=il3",
        "filter/initial=vhh_il3",
        "filter/final=vhh_il3",
      ],
    )
  processed = process_config(cfg)
  run_settings = processed["run"]
  target_settings = processed["target"]
  complex_sequences = get_sequence_from_pdb(str(fixture_path.resolve()))
  run_settings["cdr_positions"] = compute_cdr_positions(
      run_settings["cdr_lengths"], run_settings["fw_lengths"]
  )
  run_settings["starting_binder_seq"] = complex_sequences["B"]

  # Integration-only execution controls. All loss weights, thresholds, model
  # recycles, gradient merge settings, and search semantics remain inherited.
  run_settings.update(
      {
          "logits_steps": args.steps,
          "softmax_steps": 0,
          "search_steps": 0,
          "design_models": [0],
          "num_models": 1,
          "sample_models": False,
          "af_params_dir": str(args.params_dir.resolve()),
          "starting_pdb_complex": str(fixture_path.resolve()),
          "save_design_trajectory_plots": False,
          "save_design_animations": False,
      }
  )
  target_settings.update(
      {
          "target_pdb_path": str(fixture_path.resolve()),
          "target_hotspots": (
              "A23,A25,A26,A31,A40,A104"
              if args.target_residues >= 112
              else "A23,A25,A26"
          ),
      }
  )

  design_name = (
      f"phase2b_{args.backend}_t{args.target_residues}_{args.steps}step"
  )
  layout = RunLayout.create(output_root)
  design_io = IO(layout)
  step_records = []
  single_pass_records = []
  original_step = _af_design.step
  original_single = _af_design._single
  initial_params = None

  def instrumented_step(model, *step_args, **step_kwargs):
    nonlocal initial_params
    before = np.array(model._params["seq"])
    if initial_params is None:
      initial_params = before.copy()
      np.save(
          output_root
          / f"initial_seq_params_t{args.target_residues}_{args.steps}step.npy",
          initial_params,
      )
    result = original_step(model, *step_args, **step_kwargs)
    after = np.array(model._params["seq"])
    gradient = np.asarray(model.aux["grad"]["seq"])
    atom_positions = model.aux["atom_positions"]
    record = {
        "iteration": len(step_records) + 1,
        "loss": float(model.aux["loss"]),
        "loss_components": {
            key: float(value)
            for key, value in model.aux["losses"].items()
        },
        "gradient_shape": list(gradient.shape),
        "gradient_device": _tensor_devices(model.aux["grad"]["seq"]),
        "parameter_device": _tensor_devices(model._params["seq"]),
        "atom_positions_device": _tensor_devices(atom_positions),
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "parameter_update_l2": float(np.linalg.norm(after - before)),
        "parameter_update_max_abs": float(np.max(np.abs(after - before))),
        "params_changed": bool(not np.array_equal(after, before)),
        "plddt": float(np.mean(np.asarray(model.aux["plddt"]))),
        "iptm": float(np.asarray(model.aux["i_ptm"])),
        "lm_ll": float(model.aux["log"]["lm_ll"]),
        "language_model_device": str(model.ablm_model.device),
    }
    step_records.append(record)
    print(json.dumps(record), flush=True)
    return result

  def instrumented_single(model, model_params, backprop=True):
    aux = original_single(model, model_params, backprop=backprop)
    record = {
        "step": len(step_records) + 1,
        "backprop": backprop,
        "loss": float(aux["loss"]),
        "loss_device": _tensor_devices(aux["loss"]),
        "plddt_device": _tensor_devices(aux["plddt"]),
        "pae_device": _tensor_devices(aux["pae"]),
        "atom_positions_device": _tensor_devices(aux["atom_positions"]),
        "gradient_device": _tensor_devices(aux["grad"]["seq"]),
    }
    single_pass_records.append(record)
    return aux

  _af_design.step = instrumented_step
  _af_design._single = instrumented_single
  started = time.perf_counter()
  model_error = None
  try:
    model = germinal_design(
        design_name,
        run_settings,
        target_settings,
        design_io,
        seed=42,
    )
  except Exception as exc:
    model_error = f"{type(exc).__name__}: {exc}"
    raise
  finally:
    _af_design.step = original_step
    _af_design._single = original_single
  elapsed = time.perf_counter() - started
  final_params = np.asarray(model._params["seq"])
  np.save(
      output_root
      / f"final_seq_params_t{args.target_residues}_{args.steps}step.npy",
      final_params,
  )
  final_sequence = model.get_seq(get_best=False)[0]

  expected_device = "MPS:0" if args.backend == "mps" else "cpu:0"
  step_devices = {
      device
      for record in step_records
      for key in (
          "gradient_device",
          "parameter_device",
          "atom_positions_device",
      )
      for device in record[key]
  }
  checks = {
      "steps_completed": len(step_records) == args.steps,
      "finite_losses": all(
          np.isfinite(record["loss"]) for record in step_records
      ),
      "finite_gradients": all(
          record["gradient_finite"] for record in step_records
      ),
      "nonzero_gradients": all(
          record["gradient_l2"] > 0 for record in step_records
      ),
      "parameter_updates": all(
          record["params_changed"] and record["parameter_update_l2"] > 0
          for record in step_records
      ),
      "expected_device_only": step_devices == {expected_device},
      "pdb_saved": (
          layout.trajectories / "structures" / f"{design_name}.pdb"
      ).is_file(),
  }
  final_pdb = layout.trajectories / "structures" / f"{design_name}.pdb"
  report = {
      "status": "PASS" if all(checks.values()) else "FAIL",
      "backend_requested": args.backend,
      "jax_backend": jax.default_backend(),
      "device": str(jax.devices(args.backend)[0]),
      "python": sys.version.split()[0],
      "jax_version": jax.__version__,
      "fixture": {
          "path": str(fixture_path),
          "sha256": _sha256(fixture_path),
          "target_residues": args.target_residues,
          "binder_residues": 131,
          "target_hotspots": target_settings["target_hotspots"],
      },
      "scientific_settings": {
          "source_config": "vhh_il3 + il3 + vhh_il3 filters",
          "loss_weights": run_settings["weights_plddt"],
          "pae_intra": run_settings["weights_pae_intra"],
          "pae_inter": run_settings["weights_pae_inter"],
          "iptm_weight": run_settings["weights_iptm"],
          "helix_weight": run_settings["weights_helix"],
          "beta_weight": run_settings["weights_beta"],
          "rg_weight": run_settings["weights_rg"],
          "grad_merge_method": run_settings["grad_merge_method"],
          "ablm_model": run_settings["ablm_model"],
          "ablm_scale": run_settings["ablm_scale"],
          "num_recycles_design": run_settings["num_recycles_design"],
          "logits_steps": args.steps,
          "softmax_steps": 0,
          "search_steps": 0,
          "af2_models_executed": [1],
      },
      "steps": step_records,
      "single_passes": single_pass_records,
      "initial_sequence": "".join(
          np.array(list("ARNDCQEGHILKMFPSTWYV"))[
              initial_params[0].argmax(-1)
          ]
      ),
      "final_sequence": final_sequence,
      "initial_params_sha256": hashlib.sha256(
          initial_params.tobytes()
      ).hexdigest(),
      "final_params_sha256": hashlib.sha256(
          final_params.tobytes()
      ).hexdigest(),
      "initial_params_path": str(
          output_root
          / f"initial_seq_params_t{args.target_residues}_{args.steps}step.npy"
      ),
      "final_params_path": str(
          output_root
          / f"final_seq_params_t{args.target_residues}_{args.steps}step.npy"
      ),
      "checks": checks,
      "runtime_seconds": elapsed,
      "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "final_pdb": {
          "path": str(final_pdb),
          "sha256": _sha256(final_pdb) if final_pdb.is_file() else None,
      },
      "error": model_error,
  }
  report_path = (
      output_root
      / f"phase2b_design_t{args.target_residues}_{args.steps}step_report.json"
  )
  report_path.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2), flush=True)
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
