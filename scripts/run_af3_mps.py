#!/usr/bin/env python3
"""Run pinned AlphaFold 3 natively on Apple Silicon through JAX MPS.

This script is the stable subprocess boundary between Germinal's Python 3.10
environment and the dedicated AF3 Python 3.12 environment. It intentionally
uses only the interpreter that executes it plus the configured AF3 checkout.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PACKAGE_PINS = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "jax-mps": "0.10.9",
    "tokamax": "0.0.12",
}
TOOLS = ("jackhmmer", "nhmmer", "hmmalign", "hmmsearch", "hmmbuild")
DATABASE_FLAGS = {
    "small_bfd_database_path": "--small_bfd_database_path",
    "mgnify_database_path": "--mgnify_database_path",
    "uniprot_cluster_annot_database_path": "--uniprot_cluster_annot_database_path",
    "uniref90_database_path": "--uniref90_database_path",
    "ntrna_database_path": "--ntrna_database_path",
    "rfam_database_path": "--rfam_database_path",
    "rna_central_database_path": "--rna_central_database_path",
    "pdb_database_path": "--pdb_database_path",
    "seqres_database_path": "--seqres_database_path",
}


class PreflightError(RuntimeError):
    """Actionable failure before AF3 inference starts."""


def log(message: str) -> None:
    print(f"[af3-native] {message}", flush=True)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _check_checkout(source: Path, expected_revision: str) -> None:
    if not (source / "run_alphafold.py").is_file():
        raise PreflightError(
            f"AF3 checkout has no run_alphafold.py: {source}"
        )
    if not (source / "src" / "alphafold3").is_dir():
        raise PreflightError(f"AF3 checkout has no src/alphafold3: {source}")
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise PreflightError(
            f"Could not read AF3 checkout revision from {source}: "
            f"{result.stderr.strip()}"
        )
    actual = result.stdout.strip()
    if actual != expected_revision:
        raise PreflightError(
            f"AF3 revision mismatch: expected {expected_revision}, got {actual}. "
            f"Update af3_expected_revision only with an intentional validated pin."
        )
    log(f"AF3 checkout revision verified: {actual}")


def _check_model(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise PreflightError(f"AF3 model directory not found: {model_dir}")
    candidates = sorted(
        list(model_dir.glob("af3.bin.zst"))
        + list(model_dir.glob("af3.bin"))
        + list(model_dir.glob("*.bin.zst"))
    )
    if not candidates:
        raise PreflightError(
            f"No AF3 parameter archive found in {model_dir}; expected "
            "af3.bin.zst or another *.bin.zst file."
        )
    log(f"AF3 model archive found: {candidates[0]}")


def _check_tools(tools_bin_dir: Path) -> None:
    if not tools_bin_dir.is_dir():
        raise PreflightError(f"HMMER tools directory not found: {tools_bin_dir}")
    missing = [tool for tool in TOOLS if not (tools_bin_dir / tool).is_file()]
    if missing:
        raise PreflightError(
            f"HMMER tools directory {tools_bin_dir} is missing: "
            + ", ".join(missing)
        )
    log(f"HMMER tools verified: {tools_bin_dir}")


def _check_packages_and_patch(tokamax_patch: Path) -> None:
    if not tokamax_patch.is_file():
        raise PreflightError(
            f"Tokamax MPS patch not found: {tokamax_patch}"
        )
    for package, expected in PACKAGE_PINS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PreflightError(
                f"Required package {package}=={expected} is not installed in the "
                "AF3 interpreter."
            ) from exc
        if actual != expected:
            raise PreflightError(
                f"AF3 dependency drift: expected {package}=={expected}, got "
                f"{actual}. Preserve the Phase 1D dependency pin."
            )
    log("Pinned JAX, JAXLib, jax-mps, and Tokamax versions verified")

    try:
        from tokamax._src import precision  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise PreflightError(f"Could not import Tokamax precision handling: {exc}") from exc
    mps_precision = getattr(precision, "_F32_DOT_PRECISION_MAP", {}).get("mps")
    if not mps_precision:
        raise PreflightError(
            "Tokamax 0.0.12 does not have the required MPS precision mapping. "
            f"Apply the configured reversible patch {tokamax_patch}, then retry."
        )
    expected_precision = {
        precision.Precision.DEFAULT: "F32_F32_F32",
        precision.Precision.HIGH: "F32_F32_F32",
        precision.Precision.HIGHEST: "F32_F32_F32",
    }
    if dict(mps_precision) != expected_precision:
        raise PreflightError(
            f"Unexpected Tokamax MPS precision mapping: {dict(mps_precision)}"
        )
    log("Tokamax MPS float32 precision mapping verified")


def _check_mps() -> None:
    os.environ["JAX_PLATFORMS"] = "mps"
    try:
        import jax  # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise PreflightError(f"Could not import JAX MPS: {exc}") from exc

    try:
        backend = jax.default_backend()
        devices = jax.devices("mps")
    except Exception as exc:
        raise PreflightError(
            "JAX MPS is unavailable. AF3 native execution will not fall back to "
            f"CPU. Check Metal availability and jax-mps installation: {exc}"
        ) from exc
    if backend != "mps" or not devices:
        raise PreflightError(
            "JAX did not select MPS: "
            f"default_backend={backend!r}, devices={devices!r}. CPU fallback is "
            "disabled for this runner."
        )

    try:
        value = jax.jit(lambda x: jnp.sum(x * 2))(jnp.arange(8, dtype=jnp.float32))
        result = float(value)
        value.block_until_ready()
    except Exception as exc:
        raise PreflightError(f"JAX MPS device smoke test failed: {exc}") from exc
    if result != 56.0:
        raise PreflightError(f"Unexpected JAX MPS smoke-test result: {result}")
    log(f"JAX MPS selected explicitly: default_backend={backend}, device={devices[0]}")


def _database_arguments(args: argparse.Namespace) -> list[str]:
    if not args.run_data_pipeline:
        log("AF3 data pipeline disabled by configuration; inference-only run")
        return []

    command: list[str] = []
    if args.db_dir:
        db_dir = _path(args.db_dir)
        if not db_dir.is_dir():
            raise PreflightError(f"AF3 database directory not found: {db_dir}")
        command.extend(["--db_dir", str(db_dir)])
        log(f"AF3 database root configured: {db_dir}")

    if args.database_manifest:
        manifest_path = _path(args.database_manifest)
        if not manifest_path.is_file():
            raise PreflightError(
                f"AF3 database manifest not found: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PreflightError(
                f"Could not parse AF3 database manifest {manifest_path}: {exc}"
            ) from exc
        missing_keys = sorted(set(DATABASE_FLAGS) - set(manifest))
        if missing_keys:
            raise PreflightError(
                f"AF3 database manifest {manifest_path} is missing: "
                + ", ".join(missing_keys)
            )
        for key, flag in DATABASE_FLAGS.items():
            configured = Path(manifest[key]).expanduser()
            if not configured.is_absolute():
                configured = manifest_path.parent / configured
            configured = configured.resolve()
            if not configured.exists():
                raise PreflightError(
                    f"AF3 database path for {key} does not exist: {configured}"
                )
            command.extend([flag, str(configured)])
        log(f"AF3 explicit database manifest verified: {manifest_path}")

    if not command:
        raise PreflightError(
            "AF3 data pipeline requires --db-dir or --database-manifest."
        )
    return command


def _run_af3(command: list[str], env: dict[str, str], timeout: float) -> int:
    log(f"Starting upstream AF3: {command[0]}")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=None,
        stderr=None,
        text=True,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(
            f"AF3 exceeded the configured {timeout:.0f}s timeout; terminating "
            f"process group {process.pid}"
        )
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        return 124


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--af3-source", required=True)
    parser.add_argument("--expected-af3-revision", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tools-bin-dir", required=True)
    parser.add_argument("--tokamax-patch", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--db-dir")
    parser.add_argument("--database-manifest")
    parser.add_argument("--run-data-pipeline", action="store_true")
    parser.add_argument("--num-diffusion-samples", type=int, default=5)
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    if args.num_diffusion_samples < 1:
        parser.error("--num-diffusion-samples must be >= 1")
    if args.num_recycles < 1:
        parser.error("--num-recycles must be >= 1")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be > 0")
    return args


def main() -> int:
    args = parse_args()
    source = _path(args.af3_source)
    model_dir = _path(args.model_dir)
    tools_bin_dir = _path(args.tools_bin_dir)
    tokamax_patch = _path(args.tokamax_patch)
    output_dir = _path(args.output_dir)
    json_path = _path(args.json_path)

    if sys.version_info[:2] < (3, 12):
        raise PreflightError(
            f"AF3 native MPS requires Python >=3.12; got {sys.version.split()[0]}"
        )
    if sys.platform != "darwin":
        raise PreflightError(
            "The native AF3 MPS runner is supported only on macOS."
        )
    if not json_path.is_file():
        raise PreflightError(f"AF3 input JSON not found: {json_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _check_checkout(source, args.expected_af3_revision)
    _check_model(model_dir)
    _check_tools(tools_bin_dir)
    _check_packages_and_patch(tokamax_patch)
    _check_mps()
    database_args = _database_arguments(args)

    source_launcher = Path(__file__).with_name("run_af3_from_source.py")
    if not source_launcher.is_file():
        raise PreflightError(f"AF3 source launcher not found: {source_launcher}")
    af3_command = [
        sys.executable,
        str(source_launcher),
        "--af3-source",
        str(source),
        "--json_path",
        str(json_path),
        "--output_dir",
        str(output_dir),
        "--model_dir",
        str(model_dir),
        "--jax_backend=mps",
        "--flash_attention_implementation=xla",
        f"--num_diffusion_samples={args.num_diffusion_samples}",
        f"--num_recycles={args.num_recycles}",
        "--run_inference=true",
        f"--run_data_pipeline={'true' if args.run_data_pipeline else 'false'}",
        *database_args,
    ]
    env = os.environ.copy()
    source_path = str(source / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else source_path + os.pathsep + current_pythonpath
    )
    env["PATH"] = str(tools_bin_dir) + os.pathsep + env.get("PATH", "")
    env["JAX_PLATFORMS"] = "mps"

    started = time.perf_counter()
    return_code = _run_af3(af3_command, env, args.timeout_seconds)
    elapsed = time.perf_counter() - started
    if return_code:
        log(f"Upstream AF3 failed with exit code {return_code} after {elapsed:.2f}s")
    else:
        log(f"Upstream AF3 completed in {elapsed:.2f}s")
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as exc:
        log(f"PREFLIGHT FAILED: {exc}")
        raise SystemExit(2)
