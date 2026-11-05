#!/usr/bin/env python3
"""Repeat the production AbMPNN boundary, optionally after real AF3 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/porting/af3_mps_probe"
DEFAULT_RESULTS = DOCS / "results/phase2b10"
DEFAULT_SOURCE_AF3 = (
    DOCS
    / "results/phase2b9/run2/results/germinal_run/trajectories/structures"
)
DEFAULT_PDB = (
    DEFAULT_SOURCE_AF3 / "il3_nb_s1791095845.pdb"
)
DEFAULT_CONFIG = (
    DOCS / "results/phase2b9/run2/results/germinal_run/final_config.yaml"
)
DESIGN_PYTHON = REPO_ROOT / ".envs/germinal-mps-design/bin/python"
AF3_PYTHON = REPO_ROOT / ".envs/af3-mps-probe/bin/python"
AF3_RUNNER = REPO_ROOT / "scripts/run_af3_mps.py"
AF3_SOURCE = REPO_ROOT / ".envs/af3-mps-probe/share/af3-src"
AF3_MODELS = REPO_ROOT / ".envs/af3-mps-probe/share/af3-models"
AF3_TOOLS = REPO_ROOT / ".envs/af3-mps-tools/bin"
AF3_PATCH = DOCS / "patches/tokamax-0.0.12-mps-f32.patch"
AF3_REVISION = "8f8ac383afba995f5953182d39a560ffb8bbbfd3"
BOUNDARY_PROBE = REPO_ROOT / "tests/integration/phase2b8_abmpnn_boundary_probe.py"
EXPECTED_SEQUENCE_SHA256 = (
    "af247058856a4f3d1151e2a86472ebd4f9ab6f6994c68eef0077b94a6b3a9a40"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("COMMAND: " + json.dumps(command) + "\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).returncode


def _prepare_af3_input(output_root: Path) -> tuple[Path, Path]:
    source_data = (
        DEFAULT_SOURCE_AF3
        / "af3_outputs_preserved/il3_nb_s1791095845"
        / "il3_nb_s1791095845_data.json"
    )
    if not source_data.is_file():
        raise FileNotFoundError(
            f"Preserved AF3 inference input not found: {source_data}"
        )
    case_root = output_root / "af3_inference_case"
    input_root = case_root / "af3_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_data,
        input_root / "il3_nb_s1791095845.json",
    )
    output_dir = case_root / "af3_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_root / "il3_nb_s1791095845.json", output_dir


def _run_af3(
    output_root: Path, environment: dict[str, str]
) -> dict:
    input_json, output_dir = _prepare_af3_input(output_root)
    command = [
        str(AF3_PYTHON),
        str(AF3_RUNNER),
        "--af3-source",
        str(AF3_SOURCE),
        "--expected-af3-revision",
        AF3_REVISION,
        "--model-dir",
        str(AF3_MODELS),
        "--tools-bin-dir",
        str(AF3_TOOLS),
        "--tokamax-patch",
        str(AF3_PATCH),
        "--output-dir",
        str(output_dir),
        "--json-path",
        str(input_json),
        "--num-diffusion-samples",
        "5",
        "--num-recycles",
        "10",
        "--timeout-seconds",
        "7200",
    ]
    log_path = output_root / "af3_inference.log"
    started = time.perf_counter()
    returncode = _run(command, log_path, environment)
    elapsed = time.perf_counter() - started
    text = log_path.read_text(errors="replace")
    return {
        "status": "PASS" if returncode == 0 else "FAIL",
        "returncode": returncode,
        "wall_seconds": elapsed,
        "database_pipeline": False,
        "input_json": str(input_json.relative_to(REPO_ROOT)),
        "input_json_sha256": sha256(input_json),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "log_sha256": sha256(log_path),
        "mps_selected": "default_backend=mps" in text,
        "model_inference_completed": (
            "Done running 1 fold jobs." in text
        ),
        "expected_three_seed_runs": len(
            set(
                re.findall(
                    r"Running model inference with seed (\d+)",
                    text,
                )
            )
        ),
    }


def _base_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(Path("/private/tmp/phase2b10-matplotlib"))
    return env


def _worker_environment(
    jax_mps_overlay: Path | None,
    mlx_compile_mode: str | None,
) -> dict[str, str]:
    env = _base_environment()
    env["JAX_PLATFORMS"] = "mps"
    env["PYTHONHASHSEED"] = "7"
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'scripts'}:{REPO_ROOT}"
        if not pythonpath
        else f"{REPO_ROOT / 'scripts'}:{REPO_ROOT}:{pythonpath}"
    )
    if jax_mps_overlay is not None:
        env["PYTHONPATH"] = (
            f"{jax_mps_overlay.resolve()}:{env['PYTHONPATH']}"
        )
    if mlx_compile_mode is not None:
        env["MLX_COMPILE_MODE"] = mlx_compile_mode
    return env


def _run_boundary_calls(
    output_root: Path,
    calls: int,
    interval_seconds: float,
    jax_mps_overlay: Path | None,
    mlx_compile_mode: str | None,
) -> list[dict]:
    env = _worker_environment(jax_mps_overlay, mlx_compile_mode)
    call_root = output_root / "abmpnn_boundary_calls"
    call_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(1, calls + 1):
        result_path = call_root / f"call_{index:03d}.json"
        log_path = call_root / f"call_{index:03d}.log"
        command = [
            str(DESIGN_PYTHON),
            "-u",
            str(BOUNDARY_PROBE),
            "--pdb",
            str(DEFAULT_PDB),
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(result_path),
            "--seed",
            "1791095845",
            "--dalphaball",
            str(REPO_ROOT / ".envs/germinal-macos/bin/dalphaball"),
            "--initialize-pyrosetta",
        ]
        started = time.perf_counter()
        returncode = _run(command, log_path, env)
        wall_seconds = time.perf_counter() - started
        result = (
            json.loads(result_path.read_text())
            if result_path.is_file()
            else None
        )
        sequences = [] if result is None else [
            item["seq"] for item in result.get("sequences", [])
        ]
        records.append(
            {
                "call": index,
                "returncode": returncode,
                "wall_seconds": wall_seconds,
                "status": (
                    result.get("status")
                    if result is not None
                    else "FAIL"
                ),
                "result": str(result_path.relative_to(REPO_ROOT)),
                "log": str(log_path.relative_to(REPO_ROOT)),
                "sequence_count": len(sequences),
                "unique_sequence_count": len(set(sequences)),
                "sequences_sha256": hashlib.sha256(
                    json.dumps(
                        sequences, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "all_scores_finite": bool(
                    result is not None
                    and result.get("sequences")
                    and all(
                        float(item["score"]) == float(item["score"])
                        for item in result["sequences"]
                    )
                ),
            }
        )
        if interval_seconds:
            time.sleep(interval_seconds)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--af3-inference",
        action="store_true",
        help="Run real AF3 MPS inference before the AbMPNN boundary calls.",
    )
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument(
        "--jax-mps-overlay",
        type=Path,
        help="Prepend an alternate jax-mps wheel extraction to PYTHONPATH.",
    )
    parser.add_argument(
        "--mlx-compile-mode",
        choices=("default", "no_fuse"),
        help="Set MLX_COMPILE_MODE for the AbMPNN worker boundary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "post_af3_repeat_probe.json",
    )
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("--calls must be >= 1")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlx_compile_mode = (
        None
        if args.mlx_compile_mode in {None, "default"}
        else args.mlx_compile_mode
    )
    worker_environment = _worker_environment(
        args.jax_mps_overlay,
        mlx_compile_mode,
    )
    af3_environment = _base_environment()
    started = time.perf_counter()
    af3_result = (
        _run_af3(args.output.parent, af3_environment)
        if args.af3_inference
        else None
    )
    boundary_calls = _run_boundary_calls(
        args.output.parent,
        args.calls,
        args.interval_seconds,
        args.jax_mps_overlay,
        mlx_compile_mode,
    )
    passing_calls = sum(
        record["status"] == "PASS" for record in boundary_calls
    )
    exact_matching_calls = sum(
        record["sequences_sha256"] == EXPECTED_SEQUENCE_SHA256
        for record in boundary_calls
    )
    report = {
        "phase": "2B.10",
        "status": (
            "PASS"
            if (af3_result is None or af3_result["status"] == "PASS")
            and passing_calls == args.calls
            and exact_matching_calls == args.calls
            else "FAIL"
        ),
        "mode": "post_af3" if args.af3_inference else "boundary_only",
        "seed": 1791095845,
        "input_pdb": str(DEFAULT_PDB.relative_to(REPO_ROOT)),
        "input_pdb_sha256": sha256(DEFAULT_PDB),
        "config": str(DEFAULT_CONFIG.relative_to(REPO_ROOT)),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "environment": {
            key: worker_environment.get(key)
            for key in (
                "JAX_PLATFORMS",
                "PYTHONHASHSEED",
                "XLA_FLAGS",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_MPS_ASYNC_DISPATCH",
                "MLX_MAX_OPS_PER_BUFFER",
                "MLX_COMPILE_MODE",
                "MPLCONFIGDIR",
                "PYTHONPATH",
            )
        },
        "jax_mps_overlay": (
            None
            if args.jax_mps_overlay is None
            else str(args.jax_mps_overlay.resolve())
        ),
        "af3_inference": af3_result,
        "abmpnn_boundary_calls_requested": args.calls,
        "abmpnn_boundary_calls_passing": passing_calls,
        "abmpnn_boundary_calls_exact_expected_sequences": exact_matching_calls,
        "abmpnn_boundary_call_interval_seconds": args.interval_seconds,
        "abmpnn_boundary_calls": boundary_calls,
        "total_wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(
        {
            "status": report["status"],
            "mode": report["mode"],
            "af3_status": (
                None if af3_result is None else af3_result["status"]
            ),
            "passing_calls": passing_calls,
            "requested_calls": args.calls,
            "exact_matching_calls": exact_matching_calls,
        },
        indent=2,
    ))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
