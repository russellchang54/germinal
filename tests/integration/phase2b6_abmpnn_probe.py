#!/usr/bin/env python3
"""Fixed-input AbMPNN RNG and sampling probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax  # noqa: E402


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def array_digest(array: np.ndarray) -> dict:
  array = np.asarray(array)
  return {
      "shape": list(array.shape),
      "dtype": str(array.dtype),
      "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
      "minimum": float(np.min(array)),
      "maximum": float(np.max(array)),
      "mean": float(np.mean(array)),
      "finite": bool(np.isfinite(array).all()),
  }


def rng_state(module, kind: str) -> dict:
  get_state = getattr(module, "getstate", None)
  if get_state is None:
    get_state = module.get_state
  state = get_state()
  if kind == "python":
    return {
        "kind": "python",
        "version": state[0],
        "internal_sha256": sha256_bytes(
            repr(state[1]).encode("utf-8")
        ),
        "gauss_next": state[2],
    }
  return {
      "kind": "numpy",
      "state_sha256": sha256_bytes(state[1].tobytes()),
      "position": int(state[2]),
      "has_gauss": int(state[3]),
      "cached_gaussian": float(state[4]),
  }


def torch_state() -> dict:
  try:
    import torch
  except ImportError:
    return {"available": False}
  cpu = torch.random.get_rng_state()
  result = {
      "available": True,
      "cpu_sha256": sha256_bytes(cpu.numpy().tobytes()),
      "mps_available": bool(
          hasattr(torch, "mps") and torch.backends.mps.is_available()
      ),
  }
  if result["mps_available"]:
    result["mps_state_sha256"] = sha256_bytes(
        torch.mps.get_rng_state().numpy().tobytes()
    )
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--model-seed", default="none")
  parser.add_argument("--trace-key-construction", action="store_true")
  args = parser.parse_args()

  os.environ.setdefault("JAX_PLATFORMS", "mps")
  jax = configure_jax("mps")
  import yaml
  from colabdesign.mpnn.model import mk_mpnn_model
  import colabdesign.mpnn.model as mpnn_model_module
  from colabdesign.shared.utils import Key as OriginalKey

  config = yaml.safe_load(args.config.read_text())
  run_settings = config["run_settings"]
  seed = (
      None
      if args.model_seed.lower() in {"none", "null"}
      else int(args.model_seed)
  )

  key_constructions = []
  if args.trace_key_construction:
    class TracingKey(OriginalKey):
      def __init__(self, key=None, seed=None):
        before = random.getstate()
        super().__init__(key=key, seed=seed)
        key_constructions.append(
            {
                "requested_seed": seed,
                "resolved_seed": getattr(self, "seed", None),
                "python_rng_before": {
                    "sha256": sha256_bytes(
                        repr(before[1]).encode("utf-8")
                    )
                },
                "prng_key": np.asarray(self.key).tolist(),
            }
        )

    mpnn_model_module.Key = TracingKey

  initial = {
      "python": rng_state(random, "python"),
      "numpy": rng_state(np.random, "numpy"),
      "torch": torch_state(),
  }
  started = time.perf_counter()
  model = mk_mpnn_model(
      backbone_noise=run_settings["backbone_noise"],
      model_name=run_settings["model_path"],
      weights=run_settings["mpnn_weights"],
      seed=seed,
  )
  after_model_init = {
      "python": rng_state(random, "python"),
      "numpy": rng_state(np.random, "numpy"),
      "torch": torch_state(),
  }

  from germinal.filters.redesign import (
      get_sequence_from_pdb,
      hotspot_residues,
  )

  pdb_path = args.pdb.resolve()
  chains_pdb = get_sequence_from_pdb(str(pdb_path))
  binder_chain = "B" if len(chains_pdb) < 3 else "B"
  target_chain = "A"
  binder_length = len(chains_pdb[binder_chain])
  framework_positions = [
      f"{binder_chain}{position + 1}"
      for position in range(binder_length)
      if position not in run_settings["cdr_positions"]
  ]
  interface_residues = hotspot_residues(
      str(pdb_path),
      binder_chain,
      target_chain=target_chain,
      atom_distance_cutoff=run_settings["atom_distance_cutoff"],
  )
  interface_positions = [
      f"{binder_chain}{index}" for index in interface_residues
  ]
  fixed_positions = ",".join(
      set(framework_positions + interface_positions)
  )
  model.prep_inputs(
      pdb_filename=str(pdb_path),
      chain=f"{target_chain},{binder_chain}",
      fix_pos=fixed_positions,
      rm_aa=run_settings["omit_AAs"],
  )
  after_prep = {
      "python": rng_state(random, "python"),
      "numpy": rng_state(np.random, "numpy"),
      "torch": torch_state(),
  }

  output = model.sample(
      temperature=run_settings["sampling_temp"],
      num=1,
      batch=run_settings["num_seqs"],
  )
  final_rng = {
      "python": rng_state(random, "python"),
      "numpy": rng_state(np.random, "numpy"),
      "torch": torch_state(),
  }
  sequences = [str(sequence) for sequence in output["seq"]]
  token_indices = np.asarray(output["S"]).argmax(-1)
  from scipy.special import softmax

  probabilities = softmax(np.asarray(output["logits"]), axis=-1)
  report = {
      "status": "PASS",
      "python": {
          "executable": sys.executable,
          "version": platform.python_version(),
          "pid": os.getpid(),
      },
      "jax": {
          "version": jax.__version__,
          "backend": jax.default_backend(),
          "devices": [str(device) for device in jax.devices()],
      },
      "input": {
          "pdb": str(pdb_path),
          "pdb_sha256": sha256_bytes(pdb_path.read_bytes()),
          "config": str(args.config.resolve()),
          "config_sha256": sha256_bytes(args.config.read_bytes()),
          "binder_chain": binder_chain,
          "target_chain": target_chain,
          "binder_length": binder_length,
          "interface_positions": interface_positions,
          "fixed_position_count": len(fixed_positions.split(",")),
          "temperature": run_settings["sampling_temp"],
          "batch": run_settings["num_seqs"],
          "backbone_noise": run_settings["backbone_noise"],
          "model_path": run_settings["model_path"],
          "weights": run_settings["mpnn_weights"],
      },
      "seed": {
          "requested_model_seed": seed,
          "key_constructions": key_constructions,
          "random_used_without_explicit_seed": (
              seed is None and args.trace_key_construction
          ),
      },
      "rng": {
          "initial": initial,
          "after_model_init": after_model_init,
          "after_prep": after_prep,
          "final": final_rng,
      },
      "sampling": {
          "sequences": sequences,
          "scores": np.asarray(output["score"]).tolist(),
          "seqid": np.asarray(output["seqid"]).tolist(),
          "logits": array_digest(output["logits"]),
          "probability_sha256": sha256_bytes(
              np.ascontiguousarray(probabilities).tobytes()
          ),
          "selected_token_indices": array_digest(token_indices),
          "decoding_order": array_digest(output["decoding_order"]),
      },
      "runtime_seconds": time.perf_counter() - started,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
