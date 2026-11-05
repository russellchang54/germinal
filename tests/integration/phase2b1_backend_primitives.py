#!/usr/bin/env python3
"""Probe the exact JAX primitives used by Germinal beta_strand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax


def _stats(value) -> dict:
  array = np.asarray(value)
  finite = np.isfinite(array) if np.issubdtype(array.dtype, np.inexact) else None
  return {
      "shape": list(array.shape),
      "dtype": str(array.dtype),
      "device": sorted(str(device) for device in value.devices())
      if hasattr(value, "devices")
      else [],
      "min": float(np.nanmin(array))
      if np.issubdtype(array.dtype, np.inexact)
      else int(array.min()),
      "max": float(np.nanmax(array))
      if np.issubdtype(array.dtype, np.inexact)
      else int(array.max()),
      "mean": float(np.nanmean(array))
      if np.issubdtype(array.dtype, np.inexact)
      else float(array.mean()),
      "finite_count": int(finite.sum()) if finite is not None else None,
      "nan_count": int(np.isnan(array).sum())
      if np.issubdtype(array.dtype, np.inexact)
      else 0,
      "inf_count": int(np.isinf(array).sum())
      if np.issubdtype(array.dtype, np.inexact)
      else 0,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", choices=["mps", "cpu"], required=True)
  parser.add_argument("--intermediates", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--repeats", type=int, default=5)
  args = parser.parse_args()

  jax = configure_jax(args.platform)
  jnp = jax.numpy
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  source = np.load(args.intermediates)
  diagonal = jnp.asarray(source["masked_diagonal"], dtype=jnp.float32)

  top_k_count = 12
  top_k_mask = jnp.arange(diagonal.size) < top_k_count

  @jax.jit
  def sort_then_reduce(values, mask):
    sorted_values = jnp.sort(values, descending=True)
    selected = jnp.where(mask, sorted_values, 0)
    return {
        "sorted_values": sorted_values,
        "selected_values": selected,
        "selected_sum": selected.sum(-1),
        "selected_sum_axis": jnp.sum(selected, axis=-1),
        "first_twelve_sum": sorted_values[:12].sum(),
        "mask_count": jnp.sum(mask),
    }

  @jax.jit
  def sort_slice_sum(values):
    return jnp.sort(values, descending=True)[:top_k_count].sum()

  @jax.jit
  def sort_multiply_sum(values, mask):
    sorted_values = jnp.sort(values, descending=True)
    return jnp.sum(sorted_values * mask.astype(values.dtype))

  @jax.jit
  def sort_lax_reduce(values, mask):
    sorted_values = jnp.sort(values, descending=True)
    selected = jnp.where(mask, sorted_values, 0)
    return jax.lax.reduce(
        selected,
        jnp.asarray(0.0, dtype=selected.dtype),
        jax.lax.add,
        (0,),
    )

  @jax.jit
  def softmax_probe(logits, bits):
    probabilities = jax.nn.softmax(logits)
    outside_probability = (
        probabilities * bits
    ).sum(-1)
    epsilon_probability = outside_probability + 1e-8
    binary_loss = -jnp.log(epsilon_probability)
    return {
        "probabilities": probabilities,
        "outside_probability": outside_probability,
        "epsilon_probability": epsilon_probability,
        "binary_loss": binary_loss,
    }

  sorted_result = sort_then_reduce(diagonal, top_k_mask)
  jax.block_until_ready(sorted_result)
  slice_sum = sort_slice_sum(diagonal)
  multiply_sum = sort_multiply_sum(diagonal, top_k_mask)
  lax_reduce_sum = sort_lax_reduce(diagonal, top_k_mask)
  jax.block_until_ready((slice_sum, multiply_sum, lax_reduce_sum))
  compiled = sort_then_reduce.lower(diagonal, top_k_mask).compile()
  lowered_text = compiled.as_text()
  (output_dir / "sort_then_reduce_stablehlo.txt").write_text(
      lowered_text or "StableHLO text is not exposed by this backend.\n"
  )

  dgram = jnp.asarray(source["pair_loss_selected"])
  outside_bins = jnp.asarray(source["outside_bins"])
  extreme_shape = (32, 32, 64)
  inside_bins = ~outside_bins
  extreme_logits = jnp.where(
      inside_bins[None, None, :],
      jnp.float32(100.0),
      jnp.float32(-100.0),
  )
  extreme_logits = jnp.broadcast_to(extreme_logits, extreme_shape)
  extreme_softmax = softmax_probe(extreme_logits, outside_bins)
  jax.block_until_ready(extreme_softmax)

  tied_diagonal = jnp.asarray(
      [1.0] * 12 + [0.5] * 147,
      dtype=jnp.float32,
  )
  tied_sorted = jax.jit(
      lambda values: jnp.sort(values, descending=True)
  )(tied_diagonal)

  repeated = []
  for _ in range(args.repeats):
    result = sort_then_reduce(diagonal, top_k_mask)
    result["selected_sum"].block_until_ready()
    repeated.append(float(result["selected_sum"]))

  arrays = {
      "diagonal": np.asarray(jax.device_get(diagonal)),
      "extreme_logits": np.asarray(jax.device_get(extreme_logits)),
      "extreme_probabilities": np.asarray(
          jax.device_get(extreme_softmax["probabilities"])
      ),
      "extreme_outside_probability": np.asarray(
          jax.device_get(extreme_softmax["outside_probability"])
      ),
      "extreme_binary_loss": np.asarray(
          jax.device_get(extreme_softmax["binary_loss"])
      ),
      "tied_diagonal": np.asarray(jax.device_get(tied_diagonal)),
  }
  np.savez_compressed(output_dir / "arrays.npz", **arrays)

  expected_sum = float(
      np.asarray(
          jax.device_get(jnp.sort(diagonal, descending=True))[:12]
      ).sum()
  )
  report = {
      "platform": args.platform,
      "device": str(jax.devices(args.platform)[0]),
      "input_intermediates": str(args.intermediates.resolve()),
      "ordinary_model_generated_diagonal": {
          "input": _stats(diagonal),
          "sorted_values": _stats(sorted_result["sorted_values"]),
          "selected_values": _stats(sorted_result["selected_values"]),
          "mask_count": int(sorted_result["mask_count"]),
          "selected_sum_returned": float(sorted_result["selected_sum"]),
          "selected_sum_axis_returned": float(
              sorted_result["selected_sum_axis"]
          ),
          "first_twelve_sum_returned": float(
              sorted_result["first_twelve_sum"]
          ),
          "sort_slice_sum": float(slice_sum),
          "sort_multiply_mask_sum": float(multiply_sum),
          "sort_lax_reduce_sum": float(lax_reduce_sum),
          "expected_selected_sum": expected_sum,
          "selected_sum_correct": bool(
              abs(float(sorted_result["selected_sum"]) - expected_sum) < 1e-5
          ),
      },
      "extreme_logits": {
          "logits": _stats(extreme_logits),
          "probabilities": _stats(extreme_softmax["probabilities"]),
          "outside_probability": _stats(
              extreme_softmax["outside_probability"]
          ),
          "epsilon_probability": _stats(
              extreme_softmax["epsilon_probability"]
          ),
          "binary_loss": _stats(extreme_softmax["binary_loss"]),
          "expected_upper_bound": float(-np.log(np.float32(1e-8))),
      },
      "tied_values": {
          "input": _stats(tied_diagonal),
          "sort_equals_numpy": bool(
              np.array_equal(
                  np.asarray(
                      jax.device_get(tied_sorted)
                  ),
                  np.sort(np.asarray(tied_diagonal))[::-1],
              )
          ),
      },
      "repeated_selected_sums": repeated,
      "repeated_selected_sum_identical": len(set(repeated)) == 1,
  }
  (output_dir / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
