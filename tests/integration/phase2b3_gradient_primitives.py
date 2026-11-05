#!/usr/bin/env python3
"""Compare sort VJPs and lax.top_k gradients on CPU and MPS."""

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


def _sha256(array: np.ndarray) -> str:
  return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _devices(value) -> list[str]:
  return sorted(str(device) for device in value.devices())


def _gradient_record(
    gradient,
    expected: np.ndarray | None = None,
    include_values: bool = True,
) -> dict:
  array = np.asarray(gradient)
  result = {
      "shape": list(array.shape),
      "dtype": str(array.dtype),
      "device": _devices(gradient),
      "sha256": _sha256(array),
      "l2": float(np.linalg.norm(array)),
      "max_abs": float(np.abs(array).max()),
      "finite": bool(np.isfinite(array).all()),
      "nonzero_count": int(np.count_nonzero(array)),
  }
  if include_values:
    result["values"] = array.tolist()
  if expected is not None:
    delta = np.abs(array - expected)
    result["expected_match"] = bool(np.allclose(array, expected, atol=1e-6))
    result["max_abs_error"] = float(delta.max())
  return result


def _width(jnp):
  return jnp.append(0, jnp.linspace(2.3125, 21.6875, 63))


def _top_k_loss(jnp, jax, logits, positions, target_len, binder_len):
  dgram_bins = _width(jnp)
  outside_bins = jnp.logical_or(dgram_bins > 11.5, dgram_bins < 9.75)
  probabilities = jax.nn.softmax(logits)
  pair_loss = -jnp.log((outside_bins * probabilities + 1e-8).sum(-1))
  mask_1d = jnp.zeros(target_len + binder_len)
  mask_1d = mask_1d.at[positions - 1].set(1).at[positions + 1].set(1)
  mask_2d = jnp.outer(mask_1d, mask_1d)
  diagonal = jnp.diagonal(mask_2d * pair_loss, 3)
  top_k_count = positions.size // 3
  values, _ = jax.lax.top_k(diagonal, top_k_count)
  return values.sum(-1) / (top_k_count + 1e-8)


def _position_vector(jnp, total_len: int, target_len: int):
  cdr_offsets = np.asarray(
      list(range(0, 11))
      + list(range(28, 36))
      + list(range(71, 89)),
      dtype=np.int32,
  )
  positions = cdr_offsets + target_len
  positions = positions[positions < total_len - 1]
  return jnp.asarray(positions)


def _sort_vjp_probe(jax, jnp) -> dict:
  cases = [
      {
          "name": "distinct_weighted_top3_cotangent",
          "values": [3.0, 9.0, 1.0, 7.0, 5.0, 2.0],
          "cotangent": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
          "expected": [0.0, 1.0, 0.0, 2.0, 3.0, 0.0],
          "top_k": 3,
      },
      {
          "name": "distinct_sum_top3",
          "values": [3.0, 9.0, 1.0, 7.0, 5.0, 2.0],
          "cotangent": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
          "expected": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
          "top_k": 3,
      },
      {
          "name": "ties_top2",
          "values": [5.0, 4.0, 5.0, 3.0],
          "cotangent": [1.0, 1.0, 0.0, 0.0],
          "expected": [1.0, 0.0, 1.0, 0.0],
          "top_k": 2,
      },
  ]
  results = []
  for case in cases:
    values = jnp.asarray(case["values"], dtype=jnp.float32)
    cotangent = jnp.asarray(case["cotangent"], dtype=jnp.float32)
    expected = np.asarray(case["expected"], dtype=np.float32)

    def sort_vjp(input_values, output_cotangent):
      sorted_values, pullback = jax.vjp(
          lambda x: jnp.sort(x, descending=True),
          input_values,
      )
      (gradient,) = pullback(output_cotangent)
      return sorted_values, gradient

    eager_sorted, eager_gradient = sort_vjp(values, cotangent)
    jit_sorted, jit_gradient = jax.jit(sort_vjp)(values, cotangent)
    slice_gradient = jax.jit(
        jax.grad(
            lambda x: jnp.sort(x, descending=True)[: case["top_k"]].sum()
        )
    )(values)
    jax.block_until_ready(
        (
            eager_sorted,
            eager_gradient,
            jit_sorted,
            jit_gradient,
            slice_gradient,
        )
    )
    expected_slice = np.zeros(len(case["values"]), dtype=np.float32)
    selected_indices = np.argsort(
        -np.asarray(case["values"], dtype=np.float32),
        kind="stable",
    )[: case["top_k"]]
    expected_slice[selected_indices] = 1.0
    results.append(
        {
            "name": case["name"],
            "top_k": case["top_k"],
            "input": values.tolist(),
            "cotangent": cotangent.tolist(),
            "sorted_values": np.asarray(eager_sorted).tolist(),
            "sort_vjp_eager": _gradient_record(eager_gradient, expected),
            "sort_vjp_jit": _gradient_record(jit_gradient, expected),
            "sort_slice_gradient": _gradient_record(
                slice_gradient,
                expected_slice,
            ),
        }
    )
  return {
      "expected_routing_rule": (
          "Cotangents on sorted positions map back to the original indices "
          "returned by descending stable sort."
      ),
      "cases": results,
  }


def _top_k_probe(jax, jnp, actual_diagonal: np.ndarray) -> dict:
  inputs = [
      ("distinct_top3", [3.0, 9.0, 1.0, 7.0, 5.0, 2.0], 3),
      ("ties_top2", [5.0, 4.0, 5.0, 3.0], 2),
      ("all_equal_top3", [2.0] * 8, 3),
      (
          "two_rows_top3",
          [[3.0, 9.0, 1.0, 7.0, 5.0, 2.0], [0.0, 8.0, 2.0, 6.0, 4.0, 1.0]],
          3,
      ),
      ("extreme_top2", [-1.0e20, 3.0e20, 0.0, 2.0e20], 2),
      ("actual_diagonal_top12", actual_diagonal.tolist(), 12),
  ]
  results = []
  for name, raw_values, top_k_count in inputs:
    values = jnp.asarray(raw_values, dtype=jnp.float32)

    def top_k_values(input_values):
      selected, indices = jax.lax.top_k(input_values, top_k_count)
      order = jnp.arange(selected.shape[-1], dtype=selected.dtype)
      return selected, indices, (selected * (order + 1.0)).sum(-1)

    eager = jax.jit(top_k_values)(values)
    gradient_expression = jax.grad(
        lambda input_values: top_k_values(input_values)[2].sum()
    )
    eager_gradient = gradient_expression(values)
    jit_gradient = jax.jit(gradient_expression)(values)
    jax.block_until_ready((*eager, eager_gradient, jit_gradient))
    selected, indices, weighted_sum = eager
    results.append(
        {
            "name": name,
            "input_shape": list(values.shape),
            "top_k": top_k_count,
            "selected": np.asarray(selected).tolist(),
            "indices": np.asarray(indices).tolist(),
            "weighted_sum": np.asarray(weighted_sum).tolist(),
            "gradient_eager": _gradient_record(eager_gradient),
            "gradient_jit": _gradient_record(jit_gradient),
        }
    )
  return {
      "semantics": (
          "Gradient is taken through selected values only; the integer index "
          "output is not differentiated."
      ),
      "cases": results,
  }


def _actual_distogram_probe(
    jax,
    jnp,
    fixed_logits: np.ndarray,
    output_dir: Path,
) -> dict:
  positions = _position_vector(jnp, 162, 31)
  logits = jnp.asarray(fixed_logits)

  loss_fn = jax.jit(
      lambda values: _top_k_loss(jnp, jax, values, positions, 31, 131)
  )
  gradient_fn = jax.jit(
      jax.grad(
          lambda values: _top_k_loss(
              jnp,
              jax,
              values,
              positions,
              31,
              131,
          )
      )
  )
  loss = loss_fn(logits)
  gradient = gradient_fn(logits)
  jax.block_until_ready((loss, gradient))
  gradient_array = np.asarray(jax.device_get(gradient))
  np.save(output_dir / "actual_162_top_k_grad.npy", gradient_array)

  def minimal_gradient(values):
    return jax.grad(
        lambda x: jax.lax.top_k(x, 12)[0].sum()
    )(values)

  lowered = jax.jit(minimal_gradient).lower(
      jnp.arange(20, dtype=jnp.float32)
  )
  lowering_path = output_dir / "top_k_gradient_lowering.txt"
  lowering_path.write_text(
      lowered.as_text()
      or "StableHLO text is not exposed by this backend.\n"
  )
  return {
      "logits_shape": list(fixed_logits.shape),
      "logits_sha256": _sha256(fixed_logits),
      "top_k": 12,
      "loss": float(loss),
      "loss_device": _devices(loss),
      "gradient": _gradient_record(gradient, include_values=False),
      "gradient_path": str(output_dir / "actual_162_top_k_grad.npy"),
      "gradient_lowering_path": str(lowering_path),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", choices=["cpu", "mps"], required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--fixed-logits",
      type=Path,
      default=REPO_ROOT
      / "docs/porting/af3_mps_probe/results/phase2b1/processes/mps1/"
      "step2_pass06_distogram.npy",
  )
  parser.add_argument(
      "--actual-diagonal",
      type=Path,
      default=REPO_ROOT
      / "docs/porting/af3_mps_probe/results/phase2b1/"
      "isolation_mps_retry/intermediates.npz",
  )
  args = parser.parse_args()

  jax = configure_jax(args.platform)
  jnp = jax.numpy
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  fixed_logits = np.load(args.fixed_logits).astype(np.float32)
  actual_diagonal = np.load(args.actual_diagonal)["masked_diagonal"].astype(
      np.float32
  )

  report = {
      "platform": args.platform,
      "backend": jax.default_backend(),
      "device": str(jax.devices(args.platform)[0]),
      "jax_version": jax.__version__,
      "sort_vjp": _sort_vjp_probe(jax, jnp),
      "lax_top_k": _top_k_probe(jax, jnp, actual_diagonal),
      "actual_distogram": _actual_distogram_probe(
          jax,
          jnp,
          fixed_logits,
          output_dir,
      ),
  }
  (output_dir / "comparison.json").write_text(
      json.dumps(report, indent=2) + "\n"
  )
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
