#!/usr/bin/env python3
"""Compare original and prefix-slice beta_strand reductions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax


def _hash(array: np.ndarray) -> str:
  return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _devices(value) -> list[str]:
  return sorted(str(device) for device in value.devices())


def _width(distance):
  jnp = distance.jnp
  return jnp.append(0, jnp.linspace(2.3125, 21.6875, 63))


def original_loss(distance, logits, positions, target_len, binder_len):
  jax, jnp = distance.jax, distance.jnp
  dgram_bins = _width(distance)
  outside_bins = jnp.logical_or(dgram_bins > 11.5, dgram_bins < 9.75)
  px = jax.nn.softmax(logits)
  pair_loss = -jnp.log((outside_bins * px + 1e-8).sum(-1))
  mask_1d = jnp.zeros(target_len + binder_len)
  mask_1d = mask_1d.at[positions - 1].set(1).at[positions + 1].set(1)
  mask_2d = jnp.outer(mask_1d, mask_1d)
  loss_array = jnp.sort(
      jnp.diagonal(mask_2d * pair_loss, 3),
      descending=True,
  )
  top_k_mask = jnp.arange(loss_array.size) < (positions.size // 3)
  return (
      jnp.where(top_k_mask, loss_array, 0).sum(-1)
      / (top_k_mask.sum() + 1e-8)
  )


def candidate_loss(distance, logits, positions, target_len, binder_len):
  jax, jnp = distance.jax, distance.jnp
  dgram_bins = _width(distance)
  outside_bins = jnp.logical_or(dgram_bins > 11.5, dgram_bins < 9.75)
  px = jax.nn.softmax(logits)
  pair_loss = -jnp.log((outside_bins * px + 1e-8).sum(-1))
  mask_1d = jnp.zeros(target_len + binder_len)
  mask_1d = mask_1d.at[positions - 1].set(1).at[positions + 1].set(1)
  mask_2d = jnp.outer(mask_1d, mask_1d)
  loss_array = jnp.sort(
      jnp.diagonal(mask_2d * pair_loss, 3),
      descending=True,
  )
  top_k_count = positions.size // 3
  return (
      loss_array[:top_k_count].sum(-1)
      / (top_k_count + 1e-8)
  )


def _direct_reduction(jnp, values, top_k_count):
  top_k_mask = jnp.arange(values.size) < top_k_count
  original = (
      jnp.where(top_k_mask, values, 0).sum(-1)
      / (top_k_mask.sum() + 1e-8)
  )
  candidate = (
      values[:top_k_count].sum(-1)
      / (top_k_count + 1e-8)
  )
  return original, candidate


def _post_sort_gradient_probe(jax, jnp) -> dict:
  values = jnp.arange(1, 21, dtype=jnp.float32)
  top_k_count = 4
  mask = jnp.arange(values.size) < top_k_count
  index = jnp.arange(top_k_count, dtype=jnp.int32)

  def sort_sum(x):
    return jnp.sort(x, descending=True).sum()

  def masked_sort_sum(x):
    sorted_values = jnp.sort(x, descending=True)
    return jnp.where(mask, sorted_values, 0).sum()

  def sliced_sort_sum(x):
    return jnp.sort(x, descending=True)[:top_k_count].sum()

  def multiplied_sort_sum(x):
    sorted_values = jnp.sort(x, descending=True)
    return jnp.sum(sorted_values * mask.astype(x.dtype))

  def taken_sort_sum(x):
    sorted_values = jnp.sort(x, descending=True)
    return jnp.take(sorted_values, index).sum()

  expressions = {
      "sort_sum": sort_sum,
      "masked_sort_sum": masked_sort_sum,
      "sliced_sort_sum": sliced_sort_sum,
      "multiplied_sort_sum": multiplied_sort_sum,
      "taken_sort_sum": taken_sort_sum,
  }
  results = {}
  for name, expression in expressions.items():
    eager_grad = jax.grad(expression)(values)
    jit_grad = jax.jit(jax.grad(expression))(values)
    forward = expression(values)
    jax.block_until_ready((forward, eager_grad, jit_grad))
    eager_np = np.asarray(jax.device_get(eager_grad))
    jit_np = np.asarray(jax.device_get(jit_grad))
    results[name] = {
        "forward": float(forward),
        "eager_gradient": eager_np.tolist(),
        "jit_gradient": jit_np.tolist(),
        "eager_gradient_l2": float(np.linalg.norm(eager_np)),
        "jit_gradient_l2": float(np.linalg.norm(jit_np)),
        "eager_gradient_devices": _devices(eager_grad),
        "jit_gradient_devices": _devices(jit_grad),
        "eager_nonzero_count": int(np.count_nonzero(eager_np)),
        "jit_nonzero_count": int(np.count_nonzero(jit_np)),
    }
  return results


def _write_backward_lowering(
    jax,
    jnp,
    output_dir: Path,
    logits,
    positions,
    target_len: int,
    binder_len: int,
    measured_gradient: dict,
) -> Path:
  minimal_values = jnp.arange(1, 21, dtype=jnp.float32)
  minimal_top_k = 4

  def minimal_candidate(values):
    return (
        jnp.sort(values, descending=True)[:minimal_top_k].sum()
        / (minimal_top_k + 1e-8)
    )

  minimal_grad_fn = jax.grad(minimal_candidate)
  minimal_jaxpr = re.sub(
      r"\x1b\[[0-9;]*m",
      "",
      str(jax.make_jaxpr(minimal_grad_fn)(minimal_values)),
  )
  minimal_hlo = (
      jax.jit(minimal_grad_fn).lower(minimal_values).as_text()
      or "StableHLO text is not exposed by this backend.\n"
  )

  distance = type("Distance", (), {"jax": jax, "jnp": jnp})()
  candidate_grad_fn = jax.grad(
      lambda values: candidate_loss(
          distance,
          values,
          positions,
          target_len,
          binder_len,
      )
  )
  candidate_jaxpr = re.sub(
      r"\x1b\[[0-9;]*m",
      "",
      str(jax.make_jaxpr(candidate_grad_fn)(logits)),
  )
  candidate_hlo = (
      jax.jit(candidate_grad_fn).lower(logits).as_text()
      or "StableHLO text is not exposed by this backend.\n"
  )

  artifact = output_dir / "candidate_backward_lowering.txt"
  artifact.write_text(
      "\n".join(
          [
              "Phase 2B.2 candidate backward-pass capture",
              f"JAX version: {jax.__version__}",
              f"Backend: {jax.default_backend()}",
              f"Device: {jax.devices()[0]}",
              "",
              "Minimal exact forward:",
              "  jnp.sort(values, descending=True)[:4].sum() / (4 + 1e-8)",
              "",
              "Minimal reverse-mode JAXPR:",
              minimal_jaxpr,
              "",
              "Minimal StableHLO lowering of jax.jit(jax.grad(...)):",
              minimal_hlo,
              "",
              "Actual fixed-distogram candidate reverse-mode JAXPR:",
              f"Input shape: {logits.shape}",
              f"Input dtype: {logits.dtype}",
              candidate_jaxpr,
              "",
              "Actual fixed-distogram StableHLO lowering:",
              candidate_hlo,
              "",
              "Measured fixed-distogram candidate gradient:",
              json.dumps(measured_gradient, indent=2),
              "",
          ]
      )
  )
  return artifact


def _position_vector(jnp, total_len: int, target_len: int, binder_len: int):
  cdr_offsets = np.asarray(
      list(range(0, 11))
      + list(range(28, 36))
      + list(range(71, 89)),
      dtype=np.int32,
  )
  positions = cdr_offsets + target_len
  positions = positions[positions < total_len - 1]
  return jnp.asarray(positions)


def _case(
    name: str,
    logits: np.ndarray,
    positions: np.ndarray,
    target_len: int,
    binder_len: int,
    distance,
    output_dir: Path,
    check_gradient: bool = True,
) -> dict:
  jax, jnp = distance.jax, distance.jnp
  logits_jax = jnp.asarray(logits)
  positions_jax = jnp.asarray(positions)
  original_jit = jax.jit(
      lambda x: original_loss(
          distance, x, positions_jax, target_len, binder_len
      )
  )
  candidate_jit = jax.jit(
      lambda x: candidate_loss(
          distance, x, positions_jax, target_len, binder_len
      )
  )
  original_eager = original_loss(
      distance, logits_jax, positions_jax, target_len, binder_len
  )
  candidate_eager = candidate_loss(
      distance, logits_jax, positions_jax, target_len, binder_len
  )
  original_compiled = original_jit(logits_jax)
  candidate_compiled = candidate_jit(logits_jax)
  jax.block_until_ready(
      (original_eager, candidate_eager, original_compiled, candidate_compiled)
  )

  result = {
      "name": name,
      "logits_shape": list(logits.shape),
      "logits_dtype": str(logits.dtype),
      "logits_sha256": _hash(logits),
      "positions": positions.tolist(),
      "top_k_count": int(positions.size // 3),
      "forward": {
          "original_eager": float(original_eager),
          "candidate_eager": float(candidate_eager),
          "original_jit": float(original_compiled),
          "candidate_jit": float(candidate_compiled),
          "eager_abs_difference": abs(
              float(original_eager) - float(candidate_eager)
          ),
          "jit_abs_difference": abs(
              float(original_compiled) - float(candidate_compiled)
          ),
      },
      "devices": {
          "original_eager": _devices(original_eager),
          "candidate_eager": _devices(candidate_eager),
          "original_jit": _devices(original_compiled),
          "candidate_jit": _devices(candidate_compiled),
      },
      "range": {
          "original_eager_valid": 0.0 <= float(original_eager) <= 18.420681,
          "candidate_eager_valid": 0.0 <= float(candidate_eager) <= 18.420681,
          "original_jit_valid": 0.0 <= float(original_compiled) <= 18.420681,
          "candidate_jit_valid": 0.0 <= float(candidate_compiled) <= 18.420681,
      },
  }
  if check_gradient:
    original_grad_fn = jax.jit(
        jax.grad(
            lambda x: original_loss(
                distance, x, positions_jax, target_len, binder_len
            )
        )
    )
    candidate_grad_fn = jax.jit(
        jax.grad(
            lambda x: candidate_loss(
                distance, x, positions_jax, target_len, binder_len
            )
        )
    )
    original_grad = original_grad_fn(logits_jax)
    candidate_grad = candidate_grad_fn(logits_jax)
    jax.block_until_ready((original_grad, candidate_grad))
    original_grad_np = np.asarray(jax.device_get(original_grad))
    candidate_grad_np = np.asarray(jax.device_get(candidate_grad))
    np.save(output_dir / f"{name}_original_grad.npy", original_grad_np)
    np.save(output_dir / f"{name}_candidate_grad.npy", candidate_grad_np)
    delta = np.abs(original_grad_np - candidate_grad_np)
    denominator = np.maximum(
        np.maximum(np.abs(original_grad_np), np.abs(candidate_grad_np)),
        1e-12,
    )
    result["gradient"] = {
        "original_l2": float(np.linalg.norm(original_grad_np)),
        "candidate_l2": float(np.linalg.norm(candidate_grad_np)),
        "original_max_abs": float(np.abs(original_grad_np).max()),
        "candidate_max_abs": float(np.abs(candidate_grad_np).max()),
        "max_abs_difference": float(delta.max()),
        "max_relative_difference": float((delta / denominator).max()),
        "rms_difference": float(np.sqrt(np.mean(delta**2))),
        "original_device": _devices(original_grad),
        "candidate_device": _devices(candidate_grad),
        "original_finite": bool(np.isfinite(original_grad_np).all()),
        "candidate_finite": bool(np.isfinite(candidate_grad_np).all()),
    }
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", choices=["mps", "cpu"], required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--fixed-logits",
      type=Path,
      default=REPO_ROOT
      / "docs/porting/af3_mps_probe/results/phase2b1/processes/mps1/"
      "step2_pass06_distogram.npy",
  )
  args = parser.parse_args()

  jax = configure_jax(args.platform)
  jnp = jax.numpy
  distance = type("Distance", (), {"jax": jax, "jnp": jnp})()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  fixed = np.load(args.fixed_logits).astype(np.float32)
  full_positions = _position_vector(jnp, 162, 31, 131)
  cases = [
      _case(
          "actual_162",
          fixed,
          np.asarray(jax.device_get(full_positions)),
          31,
          131,
          distance,
          output_dir,
      )
  ]

  small_len = 32
  small_positions = _position_vector(jnp, small_len, 8, 24)
  small_logits = fixed[:small_len, :small_len]
  cases.append(
      _case(
          "ordinary_32",
          np.asarray(small_logits),
          np.asarray(jax.device_get(small_positions)),
          8,
          24,
          distance,
          output_dir,
      )
  )

  extreme_len = 64
  outside_bins = np.asarray(
      jax.device_get(
          jnp.logical_or(
              _width(distance) > 11.5,
              _width(distance) < 9.75,
          )
      )
  )
  extreme_logits = np.broadcast_to(
      np.where(outside_bins, -100.0, 100.0)[None, None, :],
      (extreme_len, extreme_len, 64),
  ).astype(np.float32)
  extreme_positions = _position_vector(jnp, extreme_len, 16, 48)
  cases.append(
      _case(
          "extreme_64",
          extreme_logits,
          np.asarray(jax.device_get(extreme_positions)),
          16,
          48,
          distance,
          output_dir,
      )
  )

  tiny_logits = fixed[:8, :8]
  tiny_positions = _position_vector(jnp, 8, 2, 6)
  cases.append(
      _case(
          "tiny_top_k_edge",
          tiny_logits,
          np.asarray(jax.device_get(tiny_positions)),
          2,
          6,
          distance,
          output_dir,
      )
  )

  direct_cases = [
      ("tied_reduction_159_topk12", [1.0] * 12 + [0.5] * 147, 12),
      ("empty_top_k_zero", [], 0),
      ("nonempty_top_k_zero", [1.0, 2.0, 3.0], 0),
      ("top_k_exceeds_tensor_outside_domain", [1.0, 2.0, 3.0], 5),
  ]
  for name, input_values, top_k_count in direct_cases:
    values = jnp.asarray(input_values, dtype=jnp.float32)
    direct_jit = jax.jit(lambda x: _direct_reduction(jnp, x, top_k_count))
    eager_original, eager_candidate = _direct_reduction(
        jnp, values, top_k_count
    )
    jit_original, jit_candidate = direct_jit(values)
    jax.block_until_ready(
        (
            eager_original,
            eager_candidate,
            jit_original,
            jit_candidate,
        )
    )
    cases.append(
        {
            "name": name,
            "tensor_size": int(values.size),
            "top_k_count": top_k_count,
            "forward": {
                "original_eager": float(eager_original),
                "candidate_eager": float(eager_candidate),
                "original_jit": float(jit_original),
                "candidate_jit": float(jit_candidate),
                "eager_abs_difference": abs(
                    float(eager_original) - float(eager_candidate)
                ),
                "jit_abs_difference": abs(
                    float(jit_original) - float(jit_candidate)
                ),
            },
            "devices": {
                "original_eager": _devices(eager_original),
                "candidate_eager": _devices(eager_candidate),
                "original_jit": _devices(jit_original),
                "candidate_jit": _devices(jit_candidate),
            },
            "range": {
                "original_eager_valid": (
                    0.0 <= float(eager_original) <= 18.420681
                ),
                "candidate_eager_valid": (
                    0.0 <= float(eager_candidate) <= 18.420681
                ),
                "original_jit_valid": (
                    0.0 <= float(jit_original) <= 18.420681
                ),
                "candidate_jit_valid": (
                    0.0 <= float(jit_candidate) <= 18.420681
                ),
            },
        }
    )

  backward_artifact = _write_backward_lowering(
      jax,
      jnp,
      output_dir,
      jnp.asarray(fixed),
      full_positions,
      31,
      131,
      cases[0]["gradient"],
  )

  report = {
      "platform": args.platform,
      "device": str(jax.devices(args.platform)[0]),
      "jax_version": jax.__version__,
      "cases": cases,
      "all_candidate_forward_in_range": all(
          case["range"].get("candidate_eager_valid", True)
          and case["range"].get("candidate_jit_valid", True)
          for case in cases
      ),
      "all_candidate_gradients_finite": all(
          case.get("gradient", {}).get("candidate_finite", True)
          for case in cases
      ),
      "post_sort_gradients": _post_sort_gradient_probe(jax, jnp),
      "backward_lowering_artifact": str(backward_artifact),
  }
  (output_dir / "comparison.json").write_text(
      json.dumps(report, indent=2) + "\n"
  )
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
