#!/usr/bin/env python3
"""Run and trace the fixed-seed production AF2 logits phase."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from af2_mps_compat import configure_jax  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes()
    ).hexdigest()


def load_phase2b4_helpers():
    path = (
        REPO_ROOT
        / "tests/integration/phase2b4_production_design.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase2b4_production_design", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def state_hash(tree) -> str:
    import jax

    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        array = np.asarray(jax.device_get(leaf))
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_state(path: Path, model) -> str:
    import jax

    arrays = {}
    paths = {}
    tree = {"params": model._params, "optimizer_state": model._state}
    for index, (tree_path, leaf) in enumerate(
        jax.tree_util.tree_flatten_with_path(tree)[0]
    ):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        key = f"array_{index:04d}"
        arrays[key] = np.asarray(jax.device_get(leaf))
        paths[key] = jax.tree_util.keystr(tree_path)
    np.savez_compressed(path, **arrays)
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = arrays[key]
        digest.update(paths[key].encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def array_stats(value) -> dict:
    import jax

    array = np.asarray(jax.device_get(value))
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_array(array),
        "finite": (
            bool(np.isfinite(array).all())
            if np.issubdtype(array.dtype, np.inexact)
            else True
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mps", "cpu"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1791095845)
    parser.add_argument(
        "--stop-after-logits",
        action="store_true",
        help="Stop cleanly after the unchanged 60-step logits phase.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        "/private/tmp/phase2b11-matplotlib",
    )
    jax = configure_jax(args.backend)

    import jax.numpy as jnp
    import pyrosetta as pr
    from hydra import compose, initialize_config_dir

    from colabdesign.af import design as af_design_module
    from colabdesign.af.design import AA as AF_AA
    from colabdesign.af.design import _af_design
    from germinal.design import design as design_module
    from germinal.utils import config as config_module

    helpers = load_phase2b4_helpers()
    overrides = [
        "run=vhh_il3_macos",
        "target=il3",
        "filter/initial=vhh_il3",
        "filter/final=vhh_il3",
        f"project_dir={output_root}",
        "pdb_dir=docs/porting/af3_mps_probe/results/phase2b4/run1/input",
        (
            "+starting_pdb_complex=docs/porting/af3_mps_probe/"
            "results/phase2b4/run1/input/il3_11_8_18_nb.pdb"
        ),
        (
            "af_params_dir=.envs/germinal-mps-design/share/"
            "af2-params/extracted"
        ),
        "dssp_path=.envs/germinal-macos/bin/mkdssp",
        "dalphaball_path=.envs/germinal-macos/bin/dalphaball",
        "af3_database_manifest=null",
        "af3_db_dir=/Volumes/T9/database",
        "pregenerate_seeds=true",
        "max_trajectories=1",
        "max_hallucinated_trajectories=1",
        "max_passing_designs=1",
        "show_config=false",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs"),
    ):
        cfg = compose(config_name="config", overrides=overrides)
    processed = config_module.process_config(cfg)
    run_settings = processed["run"]
    target_settings = processed["target"]
    config_module.get_jax_device = lambda: jax.devices(args.backend)
    io, run_settings = config_module.initialize_germinal_run(
        run_settings, target_settings
    )
    io.save_run_config(run_settings, target_settings)
    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        f"-holes:dalphaball {run_settings['dalphaball_path']} "
        "-corrections::beta_nov16 true -relax:default_repeats 1"
    )
    np.random.seed(1)

    trace = {
        "phase": "2B.11",
        "label": args.label,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "jax_mps_version": importlib.metadata.version("jax-mps"),
        "python": sys.version,
        "seed": args.seed,
        "input_pdb": run_settings["starting_pdb_complex"],
        "input_pdb_sha256": sha256_file(
            REPO_ROOT / run_settings["starting_pdb_complex"]
        ),
        "config": str(
            (
                Path(io.layout.root)
                / "final_config.yaml"
            ).relative_to(REPO_ROOT)
        ),
        "config_sha256": sha256_file(
            Path(io.layout.root) / "final_config.yaml"
        ),
        "schedule": {
            "logits_steps": run_settings["logits_steps"],
            "softmax_steps": run_settings["softmax_steps"],
            "search_steps": run_settings["search_steps"],
            "num_recycles_design": run_settings["num_recycles_design"],
            "sampling_temp": run_settings["sampling_temp"],
        },
        "steps": [],
        "passes": [],
        "sequence_logits": {},
        "gradients": {},
    }
    phase = {"current": "pre_design"}
    last_single = {"record": None}
    current_pcgrad = {"record": None}
    current_optax = {"record": None}

    original_mk_model = design_module.mk_afdesign_model
    original_single = _af_design._single
    original_step = _af_design.step
    original_design_logits = _af_design.design_logits
    original_pcgrad = af_design_module.pcgrad
    stop_after_logits = {"triggered": False}
    created_model = {"value": None}

    class StopAfterLogits(RuntimeError):
        pass

    def wrapped_mk_model(*model_args, **model_kwargs):
        model = original_mk_model(*model_args, **model_kwargs)
        model._args["debug"] = True
        created_model["value"] = model
        return model

    def wrapped_design_logits(model, *method_args, **method_kwargs):
        previous = phase["current"]
        phase["current"] = "logits"
        try:
            result = original_design_logits(
                model, *method_args, **method_kwargs
            )
            if args.stop_after_logits:
                stop_after_logits["triggered"] = True
                raise StopAfterLogits
            return result
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
            aux = original_single(
                model, model_params, backprop=backprop
            )
        finally:
            model.key = original_key

        debug = aux.pop("debug")
        distogram = np.asarray(
            jax.device_get(debug["outputs"]["distogram"]["logits"])
        )
        record = {
            "step": len(trace["steps"]) + 1,
            "pass": len(trace["passes"]) + 1,
            "phase": phase["current"],
            "backprop": backprop,
            "loss": float(aux["loss"]),
            "loss_components": {
                key: float(value)
                for key, value in aux["losses"].items()
                if np.asarray(jax.device_get(value)).size == 1
            },
            "distogram": {
                "shape": list(distogram.shape),
                "dtype": str(distogram.dtype),
                "sha256": sha256_array(distogram),
                "finite": bool(np.isfinite(distogram).all()),
                "nan_count": int(np.isnan(distogram).sum()),
                "inf_count": int(np.isinf(distogram).sum()),
                "minimum": float(distogram.min()),
                "maximum": float(distogram.max()),
                "mean": float(distogram.mean()),
            },
            "gradient": array_stats(aux["grad"]["seq"]),
            "plddt": array_stats(aux["plddt"]),
            "pae": array_stats(aux["pae"]),
            "atom_positions": array_stats(aux["atom_positions"]),
            "keys": [key.tolist() for key in keys],
            "key_sha256": [sha256_array(key) for key in keys],
        }
        trace["passes"].append(record)
        last_single["record"] = record
        return aux

    def wrapped_pcgrad(gradient_af2, gradient_ablm):
        combined, projected_af2, projected_iglm = original_pcgrad(
            gradient_af2, gradient_ablm
        )
        current_pcgrad["record"] = {
            "af2": array_stats(gradient_af2),
            "iglm": array_stats(gradient_ablm),
            "combined": array_stats(combined),
            "projected_af2": array_stats(projected_af2),
            "projected_iglm": array_stats(projected_iglm),
        }
        return combined, projected_af2, projected_iglm

    def wrapped_step(model, *step_args, **step_kwargs):
        iteration = len(trace["steps"])
        before_params = state_hash(model._params)
        before_optimizer = state_hash(model._state)
        current_pcgrad["record"] = None
        current_optax["record"] = None
        original_optimizer = model._optimizer

        def traced_optimizer(state, gradient, params):
            next_state, update = original_optimizer(
                state, gradient, params
            )
            current_optax["record"] = {
                "input": array_stats(gradient["seq"]),
                "update": array_stats(update["seq"]),
                "before_state_sha256": state_hash(state),
                "after_state_sha256": state_hash(next_state),
            }
            return next_state, update

        model._optimizer = traced_optimizer
        started = time.perf_counter()
        try:
            original_step(model, *step_args, **step_kwargs)
        finally:
            model._optimizer = original_optimizer

        seq_logits = np.asarray(
            jax.device_get(model._params["seq"])
        )
        gradient = np.asarray(
            jax.device_get(model.aux["grad"]["seq"])
        )
        trace["sequence_logits"][
            f"step_{iteration + 1:03d}"
        ] = seq_logits
        trace["gradients"][
            f"step_{iteration + 1:03d}"
        ] = gradient
        record = {
            "iteration": iteration + 1,
            "phase": phase["current"],
            "runtime_seconds": time.perf_counter() - started,
            "loss": float(model.aux["loss"]),
            "loss_components": {
                key: float(value)
                for key, value in model.aux["losses"].items()
            },
            "gradient": array_stats(gradient),
            "sequence_logits": array_stats(seq_logits),
            "sequence_after": "".join(
                AF_AA[seq_logits.argmax(-1)[0]]
            ),
            "parameter_sha256_before": before_params,
            "parameter_sha256_after": state_hash(model._params),
            "optimizer_state_sha256_before": before_optimizer,
            "optimizer_state_sha256_after": state_hash(model._state),
            "pcgrad": current_pcgrad["record"],
            "optax": current_optax["record"],
            "final_pass": last_single["record"],
        }
        trace["steps"].append(record)
        print(
            json.dumps(
                {
                    "step": record["iteration"],
                    "phase": record["phase"],
                    "loss": record["loss"],
                    "gradient_sha256": record["gradient"]["sha256"],
                    "parameter_sha256_after": record[
                        "parameter_sha256_after"
                    ],
                }
            ),
            flush=True,
        )

    design_module.mk_afdesign_model = wrapped_mk_model
    _af_design._single = wrapped_single
    _af_design.step = wrapped_step
    _af_design.design_logits = wrapped_design_logits
    af_design_module.pcgrad = wrapped_pcgrad
    started = time.perf_counter()
    try:
        try:
            model = design_module.germinal_design(
                f"phase2b11_{args.label}",
                run_settings,
                target_settings,
                io,
                seed=args.seed,
            )
        except StopAfterLogits:
            if not stop_after_logits["triggered"]:
                raise
            model = created_model["value"]
            if model is None:
                raise RuntimeError(
                    "Controlled logits stop occurred before model creation."
                )
    finally:
        design_module.mk_afdesign_model = original_mk_model
        _af_design._single = original_single
        _af_design.step = original_step
        _af_design.design_logits = original_design_logits
        af_design_module.pcgrad = original_pcgrad

    trace["runtime_seconds"] = time.perf_counter() - started
    trace["controlled_stop_after_logits"] = stop_after_logits[
        "triggered"
    ]
    trace["step_count"] = len(trace["steps"])
    trace["pass_count"] = len(trace["passes"])
    trace["phase_counts"] = {
        phase_name: sum(
            step["phase"] == phase_name for step in trace["steps"]
        )
        for phase_name in sorted(
            {step["phase"] for step in trace["steps"]}
        )
    }
    trace["all_steps_finite"] = all(
        np.isfinite(step["loss"])
        and step["gradient"]["finite"]
        and step["sequence_logits"]["finite"]
        and all(
            np.isfinite(value)
            for value in step["loss_components"].values()
        )
        for step in trace["steps"]
    )
    trace["all_passes_finite"] = all(
        record["distogram"]["finite"]
        and record["gradient"]["finite"]
        for record in trace["passes"]
    )
    trace["final_sequence"] = "".join(
        AF_AA[
            np.asarray(
                jax.device_get(model._params["seq"])
            ).argmax(-1)[0]
        ]
    )
    trace["best_sequence"] = "".join(
        AF_AA[
            np.asarray(
                jax.device_get(
                    model._tmp["best"]["aux"]["seq"]["pseudo"]
                )
            ).argmax(-1)[0]
        ]
    )
    trace["best_log"] = {
        key: float(
            np.asarray(jax.device_get(value)).reshape(-1)[0]
        )
        for key, value in model._tmp["best"]["log"].items()
        if np.asarray(jax.device_get(value)).size == 1
        and np.issubdtype(
            np.asarray(jax.device_get(value)).dtype, np.number
        )
    }
    trace["final_aux_log"] = {
        key: float(
            np.asarray(jax.device_get(value)).reshape(-1)[0]
        )
        for key, value in model.aux["log"].items()
        if np.asarray(jax.device_get(value)).size == 1
        and np.issubdtype(
            np.asarray(jax.device_get(value)).dtype, np.number
        )
    }
    trace["termination"] = str(model.aux["log"].get("terminate", ""))
    trace["final_parameter_sha256"] = state_hash(model._params)
    trace["final_optimizer_sha256"] = state_hash(model._state)
    trace["final_state_sha256"] = save_state(
        output_root / "final_state.npz", model
    )
    np.savez_compressed(
        output_root / "step_sequence_logits.npz",
        **trace.pop("sequence_logits"),
    )
    np.savez_compressed(
        output_root / "step_gradients.npz",
        **trace.pop("gradients"),
    )
    pdb_path = (
        Path(io.layout.trajectories)
        / "structures"
        / f"phase2b11_{args.label}.pdb"
    )
    trace["pdb"] = {
        "path": str(pdb_path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(pdb_path),
        "size_bytes": pdb_path.stat().st_size,
    }
    (output_root / "trace.json").write_text(
        json.dumps(trace, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS"
                if trace["step_count"] == 60
                and trace["all_steps_finite"]
                and trace["all_passes_finite"]
                else "FAIL",
                "trace": str(output_root / "trace.json"),
                "step_count": trace["step_count"],
                "phase_counts": trace["phase_counts"],
                "best_log": trace["best_log"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
