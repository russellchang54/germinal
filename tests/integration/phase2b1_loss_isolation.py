#!/usr/bin/env python3
"""Run the unchanged beta_strand equation on one fixed distogram tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax


def _stats(name: str, value, expected_shape=None) -> dict:
  array = np.asarray(value)
  finite = np.isfinite(array) if np.issubdtype(array.dtype, np.inexact) else None
  return {
      "name": name,
      "shape": list(array.shape),
      "dtype": str(array.dtype),
      "device": sorted(str(device) for device in value.devices())
      if hasattr(value, "devices")
      else [],
      "finite_count": int(finite.sum()) if finite is not None else None,
      "nan_count": int(np.isnan(array).sum())
      if np.issubdtype(array.dtype, np.inexact)
      else 0,
      "inf_count": int(np.isinf(array).sum())
      if np.issubdtype(array.dtype, np.inexact)
      else 0,
      "min": float(np.nanmin(array))
      if np.issubdtype(array.dtype, np.inexact)
      else int(array.min()),
      "max": float(np.nanmax(array))
      if np.issubdtype(array.dtype, np.inexact)
      else int(array.max()),
      "mean": float(np.nanmean(array))
      if np.issubdtype(array.dtype, np.inexact)
      else float(array.mean()),
      "expected_shape": list(expected_shape) if expected_shape else None,
      "shape_ok": (
          list(array.shape) == list(expected_shape)
          if expected_shape is not None
          else None
      ),
  }


def _sha256(value: np.ndarray) -> str:
  return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _calculate(jax, dgram, pos, target_len: int, binder_len: int):
  jnp = jax.numpy

  def function(logits, positions):
    dgram_bins = jnp.append(
        0,
        jnp.linspace(2.3125, 21.6875, 63),
    )
    outside_bins = jnp.logical_or(
        dgram_bins > 11.5,
        dgram_bins < 9.75,
    )
    px = jax.nn.softmax(logits)
    px_ = jax.nn.softmax(logits - 1e7 * (1 - outside_bins))
    con_loss_cat_ent = -(
        px_ * jax.nn.log_softmax(logits)
    ).sum(-1)
    con_loss_bin_ent = -jnp.log(
        (outside_bins * px + 1e-8).sum(-1)
    )
    pair_loss = jnp.where(True, con_loss_bin_ent, con_loss_cat_ent)

    mask_1d = jnp.zeros(target_len + binder_len)
    mask_1d = (
        mask_1d
        .at[positions - 1].set(1)
        .at[positions + 1].set(1)
    )
    mask_2d = jnp.outer(mask_1d, mask_1d)
    diagonal = jnp.diagonal(mask_2d * pair_loss, 3)
    sorted_values = jnp.sort(diagonal, descending=True)
    top_k_count = positions.size // 3
    top_k_mask = jnp.arange(sorted_values.size) < top_k_count
    numerator = jnp.where(top_k_mask, sorted_values, 0).sum(-1)
    denominator = top_k_mask.sum() + 1e-8
    beta = numerator / denominator
    return {
        "dgram_bins": dgram_bins,
        "outside_bins": outside_bins,
        "softmax": px,
        "masked_softmax": px_,
        "outside_probability": (outside_bins * px).sum(-1),
        "epsilon_probability": (
            (outside_bins * px + 1e-8).sum(-1)
        ),
        "pair_loss_categorical": con_loss_cat_ent,
        "pair_loss_binary": con_loss_bin_ent,
        "pair_loss_selected": pair_loss,
        "mask_1d": mask_1d,
        "mask_2d": mask_2d,
        "masked_diagonal": diagonal,
        "sorted_values": sorted_values,
        "top_k_mask": top_k_mask,
        "numerator": numerator,
        "denominator": denominator,
        "beta_strand": beta,
    }

  return function


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", choices=["mps", "cpu"], required=True)
  parser.add_argument("--distogram", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--target-len", type=int, default=31)
  parser.add_argument("--binder-len", type=int, default=131)
  parser.add_argument("--repeats", type=int, default=5)
  args = parser.parse_args()

  jax = configure_jax(args.platform)
  jnp = jax.numpy
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  source_logits = np.load(args.distogram).astype(np.float32)
  trace_json = args.distogram.parent / "trace.json"
  trace = json.loads(trace_json.read_text())
  positions = jnp.asarray(np.asarray(trace["opt_pos"], dtype=np.int32))

  function = _calculate(
      jax,
      jnp.asarray(source_logits),
      positions,
      args.target_len,
      args.binder_len,
  )
  jitted = jax.jit(function)
  jitted_result = jitted(jnp.asarray(source_logits), positions)
  jax.block_until_ready(jitted_result)
  eager_result = function(jnp.asarray(source_logits), positions)
  jax.block_until_ready(eager_result)

  arrays = {
      key: np.asarray(jax.device_get(value))
      for key, value in jitted_result.items()
  }
  np.savez_compressed(output_dir / "intermediates.npz", **arrays)

  repeated = []
  for _ in range(args.repeats):
    value = jitted(jnp.asarray(source_logits), positions)["beta_strand"]
    value.block_until_ready()
    repeated.append(float(value))

  selected_indices = np.flatnonzero(arrays["outside_bins"])
  softmax_sums = arrays["softmax"].sum(-1)
  outside = arrays["outside_probability"]
  pair_loss = arrays["pair_loss_selected"]
  selected_numerator = float(arrays["numerator"])
  denominator = float(arrays["denominator"])
  beta = float(arrays["beta_strand"])
  expected_bound = float(-np.log(np.float32(1e-8)))
  matching_trace = next(
      item
      for item in trace["single_passes"]
      if item["distogram_sha256"] == _sha256(source_logits)
  )
  stats = {
      "dgram_logits": _stats("dgram_logits", jnp.asarray(source_logits)),
      "distance_bin_values": _stats(
          "distance_bin_values", jitted_result["dgram_bins"], (64,)
      ),
      "outside_range_mask": _stats(
          "outside_range_mask", jitted_result["outside_bins"], (64,)
      ),
      "softmax_probabilities": _stats(
          "softmax_probabilities",
          jitted_result["softmax"],
          source_logits.shape,
      ),
      "masked_softmax_probabilities": _stats(
          "masked_softmax_probabilities",
          jitted_result["masked_softmax"],
          source_logits.shape,
      ),
      "outside_probability": _stats(
          "outside_probability",
          jitted_result["outside_probability"],
          source_logits.shape[:2],
      ),
      "epsilon_probability": _stats(
          "epsilon_probability",
          jitted_result["epsilon_probability"],
          source_logits.shape[:2],
      ),
      "pair_loss_categorical": _stats(
          "pair_loss_categorical",
          jitted_result["pair_loss_categorical"],
          source_logits.shape[:2],
      ),
      "pair_loss_binary": _stats(
          "pair_loss_binary",
          jitted_result["pair_loss_binary"],
          source_logits.shape[:2],
      ),
      "pair_loss_selected": _stats(
          "pair_loss_selected",
          jitted_result["pair_loss_selected"],
          source_logits.shape[:2],
      ),
      "mask_1d": _stats(
          "mask_1d",
          jitted_result["mask_1d"],
          (args.target_len + args.binder_len,),
      ),
      "mask_2d": _stats(
          "mask_2d",
          jitted_result["mask_2d"],
          (args.target_len + args.binder_len,) * 2,
      ),
      "masked_diagonal": _stats(
          "masked_diagonal",
          jitted_result["masked_diagonal"],
          (args.target_len + args.binder_len - 3,),
      ),
      "sorted_values": _stats(
          "sorted_values",
          jitted_result["sorted_values"],
          (args.target_len + args.binder_len - 3,),
      ),
      "top_k_mask": _stats(
          "top_k_mask",
          jitted_result["top_k_mask"],
          (args.target_len + args.binder_len - 3,),
      ),
  }
  report = {
      "platform": args.platform,
      "device": str(jax.devices(args.platform)[0]),
      "source_distogram": str(args.distogram.resolve()),
      "source_distogram_sha256": _sha256(source_logits),
      "statistics": stats,
      "selected_bins": {
          "indices": selected_indices.tolist(),
          "values": arrays["dgram_bins"][selected_indices].tolist(),
          "count": int(selected_indices.size),
      },
      "softmax_normalization": {
          "sum_min": float(softmax_sums.min()),
          "sum_max": float(softmax_sums.max()),
          "max_abs_deviation_from_one": float(
              np.abs(softmax_sums - 1).max()
          ),
      },
      "outside_probability": {
          "min": float(outside.min()),
          "max": float(outside.max()),
          "count_near_zero_lt_1e_8": int((outside < 1e-8).sum()),
          "minimum_after_epsilon": float(
              arrays["epsilon_probability"].min()
          ),
      },
      "pair_loss_bound": {
          "expected_upper_bound": expected_bound,
          "observed_max": float(pair_loss.max()),
          "violation_count": int((pair_loss > expected_bound + 1e-2).sum()),
      },
      "diagonal": {
          "shape": list(arrays["masked_diagonal"].shape),
          "nonzero_count": int(
              (arrays["masked_diagonal"] != 0).sum()
          ),
          "min": float(arrays["masked_diagonal"].min()),
          "max": float(arrays["masked_diagonal"].max()),
      },
      "top_k": {
          "count": int(arrays["top_k_mask"].sum()),
          "sorted_values": arrays["sorted_values"].tolist(),
          "selected_numerator": selected_numerator,
          "denominator": denominator,
          "raw_beta_strand": beta,
          "callback_weighted_value": float(
              beta * matching_trace["beta_strand_weight"]
          ),
          "actual_callback_value_from_trace": float(matching_trace["beta_strand"]),
      },
      "jit_vs_eager": {
          key: float(
              np.max(
                  np.abs(
                      np.asarray(jax.device_get(jitted_result[key]))
                      - np.asarray(jax.device_get(eager_result[key]))
                  )
              )
          )
          for key in ("beta_strand", "pair_loss_selected", "sorted_values")
      },
      "repeated_beta_within_process": repeated,
      "repeated_beta_identical": len(set(repeated)) == 1,
  }
  (output_dir / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
