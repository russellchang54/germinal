#!/usr/bin/env python3
"""Run Germinal's production design stage with Phase 2B.4 instrumentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax


def _sha256_array(array) -> str:
  import numpy as np

  return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _device(value) -> list[str]:
  return (
      sorted(str(device) for device in value.devices())
      if hasattr(value, "devices")
      else []
  )


def _device_counts(value) -> dict[str, int]:
  import jax

  counts: dict[str, int] = {}
  for leaf in jax.tree_util.tree_leaves(value):
    if hasattr(leaf, "devices"):
      for device in leaf.devices():
        name = str(device)
        counts[name] = counts.get(name, 0) + 1
    elif hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      name = "host-numpy"
      counts[name] = counts.get(name, 0) + 1
  return counts


def _array_stats(value) -> dict:
  import jax
  import numpy as np

  array = np.asarray(jax.device_get(value))
  result = {
      "shape": list(array.shape),
      "dtype": str(array.dtype),
      "device": _device(value),
      "sha256": _sha256_array(array),
      "finite": bool(np.isfinite(array).all())
      if np.issubdtype(array.dtype, np.inexact)
      else True,
      "nonzero_count": int(np.count_nonzero(array)),
  }
  if np.issubdtype(array.dtype, np.inexact):
    result.update(
        {
            "l2": float(np.linalg.norm(array)),
            "max_abs": float(np.abs(array).max()),
            "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()),
        }
    )
  return result


def _scalar_float(value) -> float:
  import numpy as np

  return float(np.asarray(value).reshape(-1)[0])


def _state_hash(tree) -> str:
  import jax
  import numpy as np

  digest = hashlib.sha256()
  flattened = jax.tree_util.tree_flatten_with_path(tree)[0]
  for path, leaf in flattened:
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
      continue
    array = np.asarray(jax.device_get(leaf))
    digest.update(jax.tree_util.keystr(path).encode())
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
  return digest.hexdigest()


def _save_state(path: Path, model) -> str:
  import jax
  import numpy as np

  arrays = {}
  paths = {}
  tree = {
      "params": model._params,
      "optimizer_state": model._state,
  }
  for index, (path_value, leaf) in enumerate(
      jax.tree_util.tree_flatten_with_path(tree)[0]
  ):
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
      continue
    key = f"array_{index:04d}"
    arrays[key] = np.asarray(jax.device_get(leaf))
    paths[key] = jax.tree_util.keystr(path_value)
  np.savez_compressed(path, **arrays)
  digest = hashlib.sha256()
  for key in sorted(arrays):
    array = arrays[key]
    digest.update(paths[key].encode())
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
  return digest.hexdigest()


def _write_minimal_complex(output_path: Path) -> None:
  import copy

  from Bio import PDB

  target = PDB.PDBParser(QUIET=True).get_structure(
      "target", REPO_ROOT / "pdbs/il3.pdb"
  )
  binder = PDB.PDBParser(QUIET=True).get_structure(
      "binder", REPO_ROOT / "pdbs/nb.pdb"
  )
  structure = PDB.Structure.Structure("phase2b4_il3_nb")
  model = PDB.Model.Model(0)
  structure.add(model)
  target_output = PDB.Chain.Chain("A")
  binder_output = PDB.Chain.Chain("B")
  model.add(target_output)
  model.add(binder_output)
  for residue in target[0]["A"]:
    target_output.add(copy.deepcopy(residue))
  for residue in binder[0]["A"]:
    binder_output.add(copy.deepcopy(residue))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  PDB.PDBIO().set_structure(structure)
  io = PDB.PDBIO()
  io.set_structure(structure)
  io.save(str(output_path))


def _configure_baseline() -> dict:
  import subprocess

  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
  ).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--short"],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
  ).stdout.splitlines()
  return {
      "git_commit": commit,
      "git_status": status,
      "design_py_sha256": _sha256_file(
          REPO_ROOT / "germinal/design/design.py"
      ),
      "phase2b3_patch_sha256": _sha256_file(
          REPO_ROOT
          / "docs/porting/af3_mps_probe/patches/beta-strand-top-k.patch"
      ),
      "vhh_il3_config_sha256": _sha256_file(
          REPO_ROOT / "configs/run/vhh_il3.yaml"
      ),
      "vhh_il3_macos_config_sha256": _sha256_file(
          REPO_ROOT / "configs/run/vhh_il3_macos.yaml"
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-label", required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--backend", choices=["mps", "cpu"], default="mps")
  parser.add_argument("--target-residues", type=int, default=112)
  parser.add_argument("--seed", type=int, default=128037)
  args = parser.parse_args()

  os.environ.setdefault(
      "MPLCONFIGDIR",
      str(REPO_ROOT / ".envs/germinal-mps-design/share/mpl"),
  )
  jax = configure_jax(args.backend)
  import jax.numpy as jnp
  import numpy as np
  from hydra import compose, initialize_config_dir

  from colabdesign.af import design as af_design_module
  from colabdesign.af.design import _af_design
  from germinal.design import design as design_module
  from germinal.utils import config as config_module
  from germinal.utils.config import process_config

  output_root = args.output_root.resolve()
  input_dir = output_root / "input"
  fixture_path = input_dir / "il3_11_8_18_nb.pdb"
  _write_minimal_complex(fixture_path)
  shutil.copy2(REPO_ROOT / "pdbs/nb.pdb", input_dir / "nb.pdb")

  overrides = [
      "run=vhh_il3_macos",
      "target=il3",
      "filter/initial=vhh_il3",
      "filter/final=vhh_il3",
      f"project_dir={output_root}",
      "results_dir=runs",
      f"experiment_name={args.run_label}",
      "run_config=",
      f"pdb_dir={input_dir}",
      (
          "af_params_dir="
          f"{REPO_ROOT / '.envs/germinal-mps-design/share/af2-params/extracted'}"
      ),
      "pregenerate_seeds=true",
      "max_trajectories=1",
      "max_hallucinated_trajectories=1",
      "max_passing_designs=1",
      "show_config=false",
  ]
  with initialize_config_dir(
      config_dir=str(REPO_ROOT / "configs"),
      version_base=None,
  ):
    cfg = compose(config_name="config", overrides=overrides)
  processed = process_config(cfg)
  run_settings = processed["run"]
  target_settings = processed["target"]

  config_module.get_jax_device = lambda: jax.devices(args.backend)
  io, run_settings = config_module.initialize_germinal_run(
      run_settings, target_settings
  )
  io.save_run_config(run_settings, target_settings)

  run_dir = Path(io.layout.root)
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "phase2b4_config.json").write_text(
      json.dumps(
          {
              "run_settings": run_settings,
              "target_settings": target_settings,
              "initial_filters": processed["filters_initial"],
              "final_filters": processed["filters_final"],
              "seed": args.seed,
              "backend": args.backend,
          },
          indent=2,
      )
      + "\n"
  )

  trace = {
      "phase": "2B.4",
      "run_label": args.run_label,
      "backend": jax.default_backend(),
      "device": str(jax.devices(args.backend)[0]),
      "jax_version": jax.__version__,
      "python": sys.version.split()[0],
      "seed": args.seed,
      "target_tokens": args.target_residues,
      "binder_tokens": 131,
      "total_tokens": args.target_residues + 131,
      "baseline": _configure_baseline(),
      "configured_schedule": {
          "logits_steps": run_settings["logits_steps"],
          "softmax_steps": run_settings["softmax_steps"],
          "search_steps": run_settings["search_steps"],
          "num_recycles_design": run_settings["num_recycles_design"],
          "design_models": run_settings["design_models"],
          "num_models": run_settings["num_models"],
          "sample_models": run_settings["sample_models"],
      },
      "passes": [],
      "steps": [],
      "host_side_operations": [
          "IgLM/PyTorch numerical execution on CPU",
          "ColabDesign NumPy PCGrad merge",
          "ColabDesign NumPy gradient normalization",
          "NumPy model selection and result/log conversion",
          "PDB serialization and host-side instrumentation hashing",
      ],
  }
  phase = {"current": "pre_design", "step_start": 0}
  last_single = {"record": None}
  current_pcgrad = {"record": None}
  current_optax = {"record": None}

  original_mk_model = design_module.mk_afdesign_model
  original_single = _af_design._single
  original_step = _af_design.step
  original_design_logits = _af_design.design_logits
  original_design_soft = _af_design.design_soft
  original_design_search = _af_design.design_pssm_semigreedy
  original_pcgrad = af_design_module.pcgrad

  def wrapped_mk_model(*model_args, **model_kwargs):
    model = original_mk_model(*model_args, **model_kwargs)
    model._args["debug"] = True
    return model

  def wrapped_design_logits(model, *method_args, **method_kwargs):
    previous = phase["current"]
    phase["current"] = "logits"
    phase["step_start"] = len(trace["steps"])
    try:
      return original_design_logits(
          model, *method_args, **method_kwargs
      )
    finally:
      phase["current"] = previous

  def wrapped_design_soft(model, *method_args, **method_kwargs):
    previous = phase["current"]
    phase["current"] = "softmax"
    phase["step_start"] = len(trace["steps"])
    try:
      return original_design_soft(model, *method_args, **method_kwargs)
    finally:
      phase["current"] = previous

  def wrapped_design_search(model, *method_args, **method_kwargs):
    previous = phase["current"]
    phase["current"] = "search"
    phase["step_start"] = len(trace["steps"])
    try:
      return original_design_search(
          model, *method_args, **method_kwargs
      )
    finally:
      phase["current"] = previous

  def wrapped_single(model, model_params, backprop=True):
    keys = []
    original_key = model.key

    def capture_key():
      key = original_key()
      keys.append(np.asarray(jax.device_get(key)))
      return key

    model.key = capture_key
    try:
      aux = original_single(model, model_params, backprop=backprop)
    finally:
      model.key = original_key

    debug = aux.pop("debug")
    outputs = debug["outputs"]
    distogram = outputs["distogram"]["logits"]
    dgram_array = np.asarray(jax.device_get(distogram))
    pass_record = {
        "step": len(trace["steps"]) + 1,
        "pass": len(trace["passes"]) + 1,
        "phase": phase["current"],
        "backprop": backprop,
        "loss": float(aux["loss"]),
        "loss_device": _device(aux["loss"]),
        "loss_components": {
            key: {
                "value": float(value),
                "device": _device(value),
            }
            for key, value in aux["losses"].items()
            if np.asarray(jax.device_get(value)).size == 1
        },
        "distogram": {
            "shape": list(dgram_array.shape),
            "dtype": str(dgram_array.dtype),
            "device": _device(distogram),
            "sha256": _sha256_array(dgram_array),
            "finite_count": int(np.isfinite(dgram_array).sum()),
            "nan_count": int(np.isnan(dgram_array).sum()),
            "inf_count": int(np.isinf(dgram_array).sum()),
            "min": float(dgram_array.min()),
            "max": float(dgram_array.max()),
            "mean": float(dgram_array.mean()),
        },
        "gradient": _array_stats(aux["grad"]["seq"]),
        "plddt_device": _device(aux["plddt"]),
        "pae_device": _device(aux["pae"]),
        "atom_positions_device": _device(aux["atom_positions"]),
        "device_counts": _device_counts(
            {
                "loss": aux["loss"],
                "losses": aux["losses"],
                "gradient": aux["grad"],
                "plddt": aux["plddt"],
                "pae": aux["pae"],
                "atom_positions": aux["atom_positions"],
                "prev": aux["prev"],
            }
        ),
        "keys": [key.tolist() for key in keys],
        "key_sha256": [
            _sha256_array(key) for key in keys
        ],
    }
    trace["passes"].append(pass_record)
    last_single["record"] = pass_record
    return aux

  def wrapped_pcgrad(gradient_af2, gradient_ablm):
    af2_device = _device(gradient_af2)
    iglm_device = _device(gradient_ablm)
    combined, projected_af2, projected_iglm = original_pcgrad(
        gradient_af2, gradient_ablm
    )
    current_pcgrad["record"] = {
        "af2": _array_stats(gradient_af2),
        "iglm": _array_stats(gradient_ablm),
        "combined": _array_stats(combined),
        "projected_af2": _array_stats(projected_af2),
        "projected_iglm": _array_stats(projected_iglm),
        "input_devices": {
            "af2": af2_device,
            "iglm": iglm_device,
        },
        "execution_platform": "host-numpy",
    }
    return combined, projected_af2, projected_iglm

  def wrapped_step(model, *step_args, **step_kwargs):
    step_number = len(trace["steps"]) + 1
    started = time.perf_counter()
    before_params_hash = _state_hash(model._params)
    before_optimizer_hash = _state_hash(model._state)
    current_pcgrad["record"] = None
    current_optax["record"] = None
    original_optimizer = model._optimizer

    def traced_optimizer(state, gradient, params):
      before_state_hash = _state_hash(state)
      input_stats = _array_stats(gradient["seq"])
      next_state, update = original_optimizer(state, gradient, params)
      current_optax["record"] = {
          "input": input_stats,
          "update": _array_stats(update["seq"]),
          "before_state_sha256": before_state_hash,
          "after_state_sha256": _state_hash(next_state),
          "device_counts": {
              "input": _device_counts(gradient),
              "update": _device_counts(update),
          },
      }
      return next_state, update

    model._optimizer = traced_optimizer
    try:
      original_step(model, *step_args, **step_kwargs)
    finally:
      model._optimizer = original_optimizer

    after_params_hash = _state_hash(model._params)
    after_optimizer_hash = _state_hash(model._state)
    loss_components = {
        key: float(value)
        for key, value in model.aux["losses"].items()
    }
    seq = np.asarray(jax.device_get(model._params["seq"]))
    step_record = {
        "iteration": step_number,
        "phase": phase["current"],
        "phase_iteration": step_number - phase["step_start"],
        "step_kwargs": {
            key: value
            for key, value in step_kwargs.items()
            if key
            in {
                "iter",
                "lr_scale",
                "num_recycles",
                "num_models",
                "sample_models",
                "backprop",
            }
        },
        "runtime_seconds": time.perf_counter() - started,
        "loss": float(model.aux["loss"]),
        "loss_components": loss_components,
        "gradient_l2": float(np.linalg.norm(model.aux["grad"]["seq"])),
        "parameter_sha256_before": before_params_hash,
        "parameter_sha256_after": after_params_hash,
        "optimizer_state_sha256_before": before_optimizer_hash,
        "optimizer_state_sha256_after": after_optimizer_hash,
        "parameter_l2_after": float(np.linalg.norm(seq)),
        "parameter_max_abs_after": float(np.abs(seq).max()),
        "pcgrad": current_pcgrad["record"],
        "optax": current_optax["record"],
        "final_pass": last_single["record"],
        "log_device_counts": _device_counts(model.aux["log"]),
    }
    trace["steps"].append(step_record)
    print(json.dumps(step_record), flush=True)

  design_module.mk_afdesign_model = wrapped_mk_model
  _af_design._single = wrapped_single
  _af_design.step = wrapped_step
  _af_design.design_logits = wrapped_design_logits
  _af_design.design_soft = wrapped_design_soft
  _af_design.design_pssm_semigreedy = wrapped_design_search
  af_design_module.pcgrad = wrapped_pcgrad
  started = time.perf_counter()
  try:
    design_name = f"phase2b4_{args.run_label}"
    model = design_module.germinal_design(
        design_name,
        run_settings,
        target_settings,
        io,
        seed=args.seed,
    )
  finally:
    design_module.mk_afdesign_model = original_mk_model
    _af_design._single = original_single
    _af_design.step = original_step
    _af_design.design_logits = original_design_logits
    _af_design.design_soft = original_design_soft
    _af_design.design_pssm_semigreedy = original_design_search
    af_design_module.pcgrad = original_pcgrad

  trace["runtime_seconds"] = time.perf_counter() - started
  trace["completed_phases"] = sorted(
      {record["phase"] for record in trace["steps"]}
      | {
          "search"
          for record in trace["passes"]
          if record["phase"] == "search"
      }
  )
  trace["step_count"] = len(trace["steps"])
  trace["pass_count"] = len(trace["passes"])
  trace["all_passes_finite"] = all(
      record["distogram"]["nan_count"] == 0
      and record["distogram"]["inf_count"] == 0
      and record["gradient"]["finite"]
      for record in trace["passes"]
  )
  trace["all_steps_finite"] = all(
      np.isfinite(record["loss"])
      and np.isfinite(record["gradient_l2"])
      and all(np.isfinite(value) for value in record["loss_components"].values())
      and (
          record["optax"] is None
          or (
              record["optax"]["input"]["finite"]
              and record["optax"]["update"]["finite"]
          )
      )
      for record in trace["steps"]
  )
  from colabdesign.af.design import AA

  trace["final_sequence"] = "".join(
      AA[
          np.asarray(jax.device_get(model._params["seq"]))
          .argmax(-1)[0]
      ]
  )
  trace["best_sequence"] = "".join(
      AA[
          np.asarray(
              jax.device_get(model._tmp["best"]["aux"]["seq"]["pseudo"])
          ).argmax(-1)[0]
      ]
  )
  trace["best_log"] = {
      key: _scalar_float(value)
      for key, value in model._tmp["best"]["log"].items()
      if np.asarray(value).size == 1
      and np.issubdtype(np.asarray(value).dtype, np.number)
  }
  trace["final_aux_log"] = {
      key: _scalar_float(value)
      for key, value in model.aux["log"].items()
      if np.asarray(value).size == 1
      and np.issubdtype(np.asarray(value).dtype, np.number)
  }
  trace["termination"] = str(model.aux["log"].get("terminate", ""))
  trace["final_state_sha256"] = _save_state(
      run_dir / "phase2b4_final_state.npz", model
  )
  trace["final_parameter_sha256"] = _state_hash(model._params)
  trace["final_optimizer_sha256"] = _state_hash(model._state)
  pdb_path = (
      Path(io.layout.trajectories)
      / "structures"
      / f"phase2b4_{args.run_label}.pdb"
  )
  trace["pdb_path"] = str(pdb_path)
  trace["pdb_sha256"] = _sha256_file(pdb_path)
  trace["pdb_size_bytes"] = pdb_path.stat().st_size

  (run_dir / "phase2b4_trace.json").write_text(
      json.dumps(trace, indent=2) + "\n"
  )
  print(
      json.dumps(
          {
              "phase2b4_complete": True,
              "trace": str(run_dir / "phase2b4_trace.json"),
              "pdb": str(pdb_path),
          }
      ),
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
