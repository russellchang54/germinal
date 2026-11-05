#!/usr/bin/env python3
"""Validate the isolated Germinal design runtime reaches JAX MPS."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax_mps

jax = configure_jax_mps()
jnp = jax.numpy


def main() -> int:
  device = jax.devices("mps")[0]
  value, gradient = jax.value_and_grad(
      lambda x: jnp.sum(jnp.tanh(x) ** 2)
  )(jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32))
  value.block_until_ready()
  gradient.block_until_ready()
  value_float = float(value)
  gradient_shape = list(gradient.shape)
  gradient_device = str(gradient.devices().pop())
  finite = bool(
      jnp.isfinite(value) and jnp.isfinite(gradient).all()
  )
  from colabdesign import clear_mem
  from germinal.design.design import germinal_design

  clear_mem()
  report = {
      "status": "PASS"
      if jax.default_backend() == "mps"
      and str(device) == "MPS:0"
      and finite
      else "FAIL",
      "python": sys.version.split()[0],
      "jax": jax.__version__,
      "backend": jax.default_backend(),
      "device": str(device),
      "value": value_float,
      "gradient_shape": gradient_shape,
      "gradient_device": gradient_device,
      "finite": finite,
      "colabdesign_germinal_import": callable(germinal_design),
  }
  output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
  rendered = json.dumps(report, indent=2) + "\n"
  if output:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
  print(rendered, end="")
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
