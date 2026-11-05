#!/usr/bin/env python3
"""Trace Germinal beta_strand and optimization state without changing equations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax


def _sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _array_device(value) -> list[str]:
  if hasattr(value, "devices"):
    return sorted(str(device) for device in value.devices())
  return []


def _stats(value) -> dict:
  array = np.asarray(value)
  result = {
      "shape": list(array.shape),
      "dtype": str(array.dtype),
      "device": _array_device(value),
  }
  if np.issubdtype(array.dtype, np.inexact):
    finite = np.isfinite(array)
    result.update(
        {
            "finite_count": int(finite.sum()),
            "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()),
            "min": float(array[finite].min()) if finite.any() else None,
            "max": float(array[finite].max()) if finite.any() else None,
            "mean": float(array[finite].mean()) if finite.any() else None,
        }
    )
  return result


def _flatten_arrays(tree) -> tuple[dict[str, np.ndarray], dict[str, str]]:
  import jax

  arrays = {}
  path_map = {}
  for index, (path, leaf) in enumerate(
      jax.tree_util.tree_flatten_with_path(tree)[0]
  ):
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      key = f"array_{index:04d}"
      arrays[key] = np.asarray(jax.device_get(leaf))
      path_map[key] = jax.tree_util.keystr(path)
  return arrays, path_map


def _save_state(path: Path, model) -> dict:
  arrays, path_map = _flatten_arrays(
      {
          "params": model._params,
          "inputs": model._inputs,
          "optimizer_state": model._state,
      }
  )
  np.savez_compressed(path, **arrays)
  digest = hashlib.sha256()
  for key in sorted(arrays):
    array = arrays[key]
    digest.update(key.encode())
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.asarray(array).tobytes())
  return {
      "path": str(path),
      "sha256": digest.hexdigest(),
      "array_count": len(arrays),
      "path_map": path_map,
      "optimizer_state_type": type(model._state).__name__,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", choices=["mps", "cpu"], required=True)
  parser.add_argument(
      "--beta-variant",
      choices=["original", "candidate", "top_k"],
      default="original",
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--initial-params",
      type=Path,
      default=REPO_ROOT
      / "docs/porting/af3_mps_probe/results/phase2b/mps_initial_params_1step.npy",
  )
  parser.add_argument("--steps", type=int, default=2)
  parser.add_argument("--target-residues", type=int, default=31)
  parser.add_argument("--gradient-decomposition", action="store_true")
  args = parser.parse_args()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  jax = configure_jax(args.backend)
  import jax.numpy as jnp
  from hydra import compose, initialize_config_dir

  from colabdesign.af import design as af_design_module
  from colabdesign.af.design import _af_design
  from germinal.design import design as design_module
  from germinal.utils.config import process_config
  from germinal.utils.io import IO, RunLayout
  from germinal.utils.utils import compute_cdr_positions, get_sequence_from_pdb
  from tests.integration.phase2b_design_probe import _write_minimal_complex

  fixture = (
      REPO_ROOT
      / "docs/porting/af3_mps_probe/results/phase2b/input"
      / f"il3_{args.target_residues}_nb_131.pdb"
  )
  _write_minimal_complex(
      REPO_ROOT / "pdbs/il3.pdb",
      REPO_ROOT / "pdbs/nb.pdb",
      fixture,
      args.target_residues,
  )
  common_initial = np.load(args.initial_params)

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
  complex_sequences = get_sequence_from_pdb(str(fixture))
  run_settings.update(
      {
          "logits_steps": args.steps,
          "softmax_steps": 0,
          "search_steps": 0,
          "design_models": [0],
          "num_models": 1,
          "sample_models": False,
          "af_params_dir": str(
              (
                  REPO_ROOT
                  / ".envs/germinal-mps-design/share/af2-params/minimal"
              ).resolve()
          ),
          "starting_pdb_complex": str(fixture),
          "cdr_positions": compute_cdr_positions(
              run_settings["cdr_lengths"], run_settings["fw_lengths"]
          ),
          "starting_binder_seq": complex_sequences["B"],
          "save_design_trajectory_plots": False,
          "save_design_animations": False,
      }
  )
  target_settings.update(
      {
          "target_pdb_path": str(fixture),
          "target_hotspots": (
              "A23,A25,A26,A31,A40,A104"
              if args.target_residues >= 112
              else "A23,A25,A26"
          ),
      }
  )

  run_root = output_dir / "run"
  layout = RunLayout.create(run_root)
  design_io = IO(layout)
  trace = {
      "backend_requested": args.backend,
      "beta_variant": args.beta_variant,
      "python": sys.version.split()[0],
      "jax_version": jax.__version__,
      "backend": jax.default_backend(),
      "device": str(jax.devices(args.backend)[0]),
      "initial_params_source": str(args.initial_params.resolve()),
      "initial_params_sha256": _sha256_bytes(common_initial.tobytes()),
      "target_residues": args.target_residues,
      "steps": [],
      "single_passes": [],
      "state_snapshots": [],
  }
  original_mk_model = design_module.mk_afdesign_model
  original_add_beta_strand_loss = design_module.add_beta_strand_loss
  original_step = _af_design.step
  original_single = _af_design._single
  original_pcgrad = af_design_module.pcgrad
  initialized = {"done": False}
  active_step = {"value": 1}

  def wrapped_mk_model(*model_args, **model_kwargs):
    model = original_mk_model(*model_args, **model_kwargs)
    model._args["debug"] = True

    def load_common_initial(af_model):
      if initialized["done"]:
        return
      if af_model._params["seq"].shape != common_initial.shape:
        raise ValueError(
            "Serialized initial parameter shape does not match AF2 model: "
            f"{common_initial.shape} vs {af_model._params['seq'].shape}"
        )
      af_model._params["seq"] = jnp.asarray(common_initial)
      initialized["done"] = True

    model._callbacks["design"]["pre"].append(load_common_initial)
    return model

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
    dgram = outputs["distogram"]["logits"]
    pass_index = len(trace["single_passes"]) + 1
    stem = f"step{active_step['value']}_pass{pass_index:02d}"
    dgram_path = output_dir / f"{stem}_distogram.npy"
    np.save(dgram_path, np.asarray(jax.device_get(dgram)))
    prev_arrays, prev_paths = _flatten_arrays(aux["prev"])
    prev_path = output_dir / f"{stem}_recycle.npz"
    np.savez_compressed(prev_path, **prev_arrays)
    record = {
        "step": active_step["value"],
        "pass": pass_index,
        "backprop": backprop,
        "loss": float(aux["loss"]),
        "loss_device": _array_device(aux["loss"]),
        "loss_components": {
            name: {
                "value": float(value),
                "device": _array_device(value),
            }
            for name, value in aux["losses"].items()
            if np.asarray(jax.device_get(value)).size == 1
        },
        "beta_strand": float(aux["losses"]["beta_strand"]),
        "beta_strand_device": _array_device(aux["losses"]["beta_strand"]),
        "beta_strand_weight": float(model.opt["weights"]["beta_strand"]),
        "weighted_beta_contribution": float(
            aux["losses"]["beta_strand"]
            * model.opt["weights"]["beta_strand"]
        ),
        "distogram": _stats(dgram),
        "distogram_path": str(dgram_path),
        "distogram_sha256": _sha256_bytes(
            np.asarray(jax.device_get(dgram)).tobytes()
        ),
        "recycle_path": str(prev_path),
        "recycle_path_map": prev_paths,
        "keys": [key.tolist() for key in keys],
        "key_sha256": [
            _sha256_bytes(key.tobytes()) for key in keys
        ],
        "plddt_device": _array_device(aux["plddt"]),
        "pae_device": _array_device(aux["pae"]),
        "atom_positions_device": _array_device(aux["atom_positions"]),
        "gradient_device": _array_device(aux["grad"]["seq"]),
    }
    trace["single_passes"].append(record)
    print(json.dumps(record), flush=True)
    return aux

  def wrapped_step(model, *step_args, **step_kwargs):
    step_number = len(trace["steps"]) + 1
    active_step["value"] = step_number
    if not initialized["done"]:
      if model._params["seq"].shape != common_initial.shape:
        raise ValueError(
            "Serialized initial parameter shape does not match AF2 model: "
            f"{common_initial.shape} vs {model._params['seq'].shape}"
        )
      model._params["seq"] = jnp.asarray(common_initial)
      initialized["done"] = True
    before = _save_state(
        output_dir / f"state_step{step_number}_before.npz", model
    )
    if step_number == 2 and args.gradient_decomposition:
      saved_inputs = model._inputs
      saved_opt = model.opt
      saved_aux = getattr(model, "aux", None)
      saved_weights = copy.deepcopy(model.opt["weights"])
      key_owner = getattr(model.key, "__self__", None)
      saved_key = (
          np.asarray(key_owner.key).copy()
          if key_owner is not None and hasattr(key_owner, "key")
          else None
      )
      inputs_template = copy.deepcopy(saved_inputs)

      def gradient_run(name, weights):
        model._inputs = copy.deepcopy(inputs_template)
        model.opt["weights"] = copy.deepcopy(weights)
        if key_owner is not None and saved_key is not None:
          key_owner.key = saved_key.copy()
        model.run(
            num_recycles=run_settings["num_recycles_design"],
            num_models=1,
            sample_models=False,
            models=[0],
            backprop=True,
        )
        grad = model.aux["grad"]["seq"]
        grad_array = np.asarray(jax.device_get(grad))
        path = output_dir / f"gradient_{name}.npy"
        np.save(path, grad_array)
        return {
            "path": str(path),
            "sha256": _sha256_bytes(grad_array.tobytes()),
            "shape": list(grad_array.shape),
            "dtype": str(grad_array.dtype),
            "device": _array_device(grad),
            "finite": bool(np.isfinite(grad_array).all()),
            "nonzero_count": int(np.count_nonzero(grad_array)),
            "l2": float(np.linalg.norm(grad_array)),
            "max_abs": float(np.abs(grad_array).max()),
        }

      zero_weights = {key: 0.0 for key in saved_weights}
      beta_weights = dict(zero_weights)
      beta_weights["beta_strand"] = saved_weights["beta_strand"]
      nonbeta_weights = dict(saved_weights)
      nonbeta_weights["beta_strand"] = 0.0
      temporary_single = _af_design._single

      def clean_single(inner_model, params, backprop=True):
        aux = original_single(inner_model, params, backprop=backprop)
        aux.pop("debug", None)
        return aux

      _af_design._single = clean_single
      try:
        beta_gradient = gradient_run("beta_only", beta_weights)
        nonbeta_gradient = gradient_run("nonbeta", nonbeta_weights)
        af2_gradient = gradient_run("af2_total", saved_weights)
      finally:
        _af_design._single = temporary_single
        model._inputs = saved_inputs
        model.opt = saved_opt
        model.opt["weights"] = saved_weights
        model.aux = saved_aux
        if key_owner is not None and saved_key is not None:
          key_owner.key = saved_key.copy()

      beta_model_grad = np.load(beta_gradient["path"])
      lm_gradient, lm_ll = model.ablm_model.get_ablm_grad(
          model.aux["seq"], method=None
      )
      lm_gradient_array = np.asarray(jax.device_get(lm_gradient))
      lm_gradient_path = output_dir / "gradient_iglm_raw.npy"
      np.save(lm_gradient_path, lm_gradient_array)
      trace["gradient_decomposition"] = {
          "beta_only": beta_gradient,
          "nonbeta_germinal_and_af2": nonbeta_gradient,
          "af2_total": af2_gradient,
          "beta_gradient_nonzero": bool(np.any(beta_model_grad != 0)),
          "iglm": {
              "l2": float(np.linalg.norm(lm_gradient_array)),
              "max_abs": float(np.abs(lm_gradient_array).max()),
              "finite": bool(np.isfinite(lm_gradient_array).all()),
              "nonzero_count": int(np.count_nonzero(lm_gradient_array)),
              "sha256": _sha256_bytes(lm_gradient_array.tobytes()),
              "path": str(lm_gradient_path),
              "log_likelihood": float(lm_ll),
              "device": str(model.ablm_model.device),
          },
      }

    original_optimizer = getattr(model, "_optimizer", None)
    if step_number == 2 and original_optimizer is not None:
      def traced_optimizer(state, gradient, params):
        input_array = np.asarray(jax.device_get(gradient["seq"]))
        input_device = _array_device(gradient["seq"])
        next_state, update = original_optimizer(state, gradient, params)
        update_array = np.asarray(jax.device_get(update["seq"]))
        trace.setdefault("actual_gradient_boundaries", {})["optax"] = {
            "input_l2": float(np.linalg.norm(input_array)),
            "input_max_abs": float(np.abs(input_array).max()),
            "input_device": input_device,
            "output_l2": float(np.linalg.norm(update_array)),
            "output_max_abs": float(np.abs(update_array).max()),
            "output_device": _array_device(update["seq"]),
        }
        return next_state, update

      model._optimizer = traced_optimizer
    result = original_step(model, *step_args, **step_kwargs)
    if step_number == 2 and original_optimizer is not None:
      model._optimizer = original_optimizer
    after = _save_state(
        output_dir / f"state_step{step_number}_after.npz", model
    )
    trace["state_snapshots"].extend(
        [
            {"boundary": f"step{step_number}_before", **before},
            {"boundary": f"step{step_number}_after", **after},
        ]
    )
    step_record = {
        "iteration": step_number,
        "loss": float(model.aux["loss"]),
        "beta_strand": float(model.aux["losses"]["beta_strand"]),
        "gradient_l2": float(np.linalg.norm(np.asarray(model.aux["grad"]["seq"]))),
        "before_state_sha256": before["sha256"],
        "after_state_sha256": after["sha256"],
        "opt_pos": np.asarray(model.opt["pos"]).tolist(),
        "optimizer_update": {
            "l2": float(
                np.linalg.norm(np.asarray(model.aux["grad"]["seq"]))
            ),
            "max_abs": float(
                np.abs(np.asarray(model.aux["grad"]["seq"])).max()
            ),
            "device": _array_device(model.aux["grad"]["seq"]),
        },
        "post_pcgrad_logged": {
            "l2": float(np.linalg.norm(model.aux["log"]["total_grad"])),
            "max_abs": float(np.abs(model.aux["log"]["total_grad"]).max()),
        },
    }
    trace["steps"].append(step_record)
    return result

  def traced_pcgrad(gradient_af2, gradient_ablm):
    if active_step["value"] == 2:
      g1 = np.asarray(jax.device_get(gradient_af2))
      g2 = np.asarray(jax.device_get(gradient_ablm))
      g1_device = _array_device(gradient_af2)
      g2_device = _array_device(gradient_ablm)
      combined, projected_af2, projected_ablm = original_pcgrad(
          gradient_af2, gradient_ablm
      )
      combined_array = np.asarray(jax.device_get(combined))
      projected_af2_array = np.asarray(jax.device_get(projected_af2))
      projected_ablm_array = np.asarray(jax.device_get(projected_ablm))
      np.save(output_dir / "gradient_pre_pcgrad_af2.npy", g1)
      np.save(output_dir / "gradient_pre_pcgrad_iglm.npy", g2)
      np.save(output_dir / "gradient_post_pcgrad.npy", combined_array)
      np.save(output_dir / "gradient_projected_af2.npy", projected_af2_array)
      np.save(
          output_dir / "gradient_projected_iglm.npy",
          projected_ablm_array,
      )
      trace.setdefault("actual_gradient_boundaries", {})["pcgrad"] = {
          "af2": {
              "l2": float(np.linalg.norm(g1)),
              "max_abs": float(np.abs(g1).max()),
              "device": g1_device,
              "sha256": _sha256_bytes(g1.tobytes()),
              "nonzero_count": int(np.count_nonzero(g1)),
          },
          "iglm": {
              "l2": float(np.linalg.norm(g2)),
              "max_abs": float(np.abs(g2).max()),
              "device": g2_device,
              "sha256": _sha256_bytes(g2.tobytes()),
              "nonzero_count": int(np.count_nonzero(g2)),
          },
          "combined": {
              "l2": float(np.linalg.norm(combined_array)),
              "max_abs": float(np.abs(combined_array).max()),
              "device": _array_device(combined),
              "sha256": _sha256_bytes(combined_array.tobytes()),
              "nonzero_count": int(np.count_nonzero(combined_array)),
          },
      }
      return combined, projected_af2, projected_ablm
    return original_pcgrad(gradient_af2, gradient_ablm)

  design_module.mk_afdesign_model = wrapped_mk_model
  af_design_module.pcgrad = traced_pcgrad
  if args.beta_variant in {"candidate", "top_k"}:
    def replacement_add_beta_strand_loss(model, weight=0):
      def replacement_beta_strand_loss(inputs, outputs):
        dgram = outputs["distogram"]["logits"]
        dgram_bins = design_module.get_dgram_bins(outputs)
        bins = jnp.logical_or(dgram_bins > 11.5, dgram_bins < 9.75)
        px = jax.nn.softmax(dgram)
        pair_loss = -jnp.log(
            (bins * px + 1e-8).sum(-1)
        )
        pos = jnp.array(model.opt["pos"])
        mask_1d = jnp.zeros(model._target_len + model._binder_len)
        mask_1d = (
            mask_1d
            .at[pos - 1].set(1)
            .at[pos + 1].set(1)
        )
        mask_2d = jnp.outer(mask_1d, mask_1d)
        diagonal = jnp.diagonal(mask_2d * pair_loss, 3)
        top_k_count = pos.size // 3
        if args.beta_variant == "top_k":
          selected_values, _ = jax.lax.top_k(diagonal, top_k_count)
        else:
          selected_values = jnp.sort(diagonal, descending=True)[:top_k_count]
        beta_loss = (
            selected_values.sum(-1)
            / (top_k_count + 1e-8)
        )
        return {"beta_strand": beta_loss}

      model._callbacks["model"]["loss"].append(
          replacement_beta_strand_loss
      )
      model.opt["weights"]["beta_strand"] = weight

    design_module.add_beta_strand_loss = replacement_add_beta_strand_loss
  _af_design.step = wrapped_step
  _af_design._single = wrapped_single
  started = time.perf_counter()
  try:
    design_module.germinal_design(
        "phase2b1_trace",
        run_settings,
        target_settings,
        design_io,
        seed=42,
    )
  finally:
    design_module.mk_afdesign_model = original_mk_model
    design_module.add_beta_strand_loss = original_add_beta_strand_loss
    af_design_module.pcgrad = original_pcgrad
    _af_design.step = original_step
    _af_design._single = original_single
  trace["runtime_seconds"] = time.perf_counter() - started
  trace["opt_pos"] = np.asarray(
      design_module.germinal_design._af_model.opt["pos"]
  ).tolist()
  trace_path = output_dir / "trace.json"
  trace_path.write_text(json.dumps(trace, indent=2) + "\n")
  print(json.dumps(trace, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
