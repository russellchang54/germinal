"""Compatibility setup for running legacy ColabDesign/AF2 code on jax-mps."""

from __future__ import annotations

import os
import functools
import sys
import types


def configure_jax(platform: str = "mps"):
  """Selects an explicit JAX backend and restores legacy ColabDesign aliases."""
  if platform not in {"mps", "cpu"}:
    raise ValueError(f"Unsupported experimental JAX platform: {platform!r}")
  if platform == "mps" and sys.platform != "darwin":
    raise RuntimeError("The AF2 MPS experiment requires macOS.")

  # ColabDesign sets this after importing JAX. Set it before JAX initializes so
  # the behavior is deterministic while leaving the AF2 math unchanged.
  os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
  os.environ["JAX_PLATFORMS"] = platform

  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

  if jax.default_backend() != platform:
    raise RuntimeError(
        f"JAX backend {platform!r} was required but the active backend is "
        f"{jax.default_backend()!r}. Silent backend fallback is disabled."
    )
  devices = jax.devices(platform)
  if not devices:
    raise RuntimeError(f"JAX reported no {platform!r} devices.")

  if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
  if not hasattr(jax, "tree_flatten"):
    jax.tree_flatten = jax.tree_util.tree_flatten
  if not hasattr(jax, "tree_unflatten"):
    jax.tree_unflatten = jax.tree_util.tree_unflatten
  if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves

  if not hasattr(jax, "util"):
    compatibility_util = types.ModuleType("jax.util")

    def wraps(fun, docstr=None, **_):
      def decorate(wrapper):
        wrapped = functools.wraps(fun)(wrapper)
        if docstr is not None:
          wrapped.__doc__ = docstr
        return wrapped
      return decorate

    compatibility_util.wraps = wraps
    jax.util = compatibility_util
    sys.modules["jax.util"] = compatibility_util

  original_clip = jnp.clip

  def clip_compat(a, a_min=None, a_max=None, *, min=None, max=None):
    """Preserves NumPy-style clip keywords removed from current JAX."""
    return original_clip(
        a,
        min=a_min if min is None else min,
        max=a_max if max is None else max,
    )

  jnp.clip = clip_compat

  if not hasattr(jax.lib, "xla_bridge"):
    from jax.extend import backend as jax_backend  # pylint: disable=import-outside-toplevel
    compatibility_module = types.ModuleType("jax.lib.xla_bridge")
    compatibility_module.get_backend = jax_backend.get_backend
    jax.lib.xla_bridge = compatibility_module
    sys.modules["jax.lib.xla_bridge"] = compatibility_module

  value = jax.jit(lambda x: jnp.sum(x * x))(jnp.arange(8, dtype=jnp.float32))
  value.block_until_ready()
  if float(value) != 140.0:
    raise RuntimeError(
        f"Unexpected {platform} preflight result: {float(value)}"
    )
  return jax


def configure_jax_mps():
  """Backward-compatible MPS-only entry point for the preflight script."""
  return configure_jax("mps")
