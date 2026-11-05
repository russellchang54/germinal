"""
Run AlphaFold 3 for antibody structure prediction.

This script is based on the AlphaFold 3  from the AlphaFold 3 repository:
https://github.com/deepmind/alphafold/

Attribution:
If you use this code, the AlphaFold 3 model, or outputs produced by it in your research, please cite:

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630(8016), 493–500. https://doi.org/10.1038/s41586-024-07487-w

Copyright 2024 DeepMind Technologies Limited.

License:
- The AlphaFold 3 source code is licensed under the Creative Commons Attribution-Non-Commercial ShareAlike International License, Version 4.0 (CC-BY-NC-SA 4.0). See: https://github.com/google-deepmind/alphafold3/blob/main/LICENSE
- The AlphaFold 3 model parameters are subject to the AlphaFold 3 Model Parameters Terms of Use: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

You may not use this file or the model parameters except in compliance with these terms.
"""

import subprocess
import argparse
import tempfile
import os
import json
import platform
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union, List
from colabfold.colabfold import run_mmseqs2
from Bio import PDB
from concurrent.futures import ProcessPoolExecutor, TimeoutError


AF3_EXPECTED_REVISION = "8f8ac383afba995f5953182d39a560ffb8bbbfd3"
AF3_REPO_ROOT = Path(__file__).resolve().parents[2]


class AF3ConfigurationError(ValueError):
    """Raised when the selected AF3 execution backend is incompletely configured."""


class AF3ExecutionError(subprocess.CalledProcessError):
    """AF3 subprocess failure with the complete log location attached."""

    def __init__(self, returncode, cmd, output=None, log_path=None):
        super().__init__(returncode, cmd, output=output, stderr=output)
        self.log_path = str(log_path) if log_path is not None else None

    def __str__(self):
        detail = super().__str__()
        if self.log_path:
            return f"{detail}. Full AF3 subprocess log: {self.log_path}"
        return detail


class AF3TimeoutError(subprocess.TimeoutExpired):
    """AF3 subprocess timeout with the complete log location attached."""

    def __init__(self, cmd, timeout, log_path=None):
        super().__init__(cmd, timeout)
        self.log_path = str(log_path) if log_path is not None else None

    def __str__(self):
        detail = super().__str__()
        if self.log_path:
            return f"{detail}. Full AF3 subprocess log: {self.log_path}"
        return detail


def _expanded_path(value: str) -> str:
    # Preserve the final symlink for virtual-environment interpreters. Resolving
    # .envs/<name>/bin/python would select the base Python and discard its
    # site-packages, defeating the separate-environment boundary.
    return os.path.abspath(os.path.expanduser(value))


def _required_setting(run_settings: dict, key: str, backend: str) -> str:
    value = run_settings.get(key)
    if value in (None, ""):
        raise AF3ConfigurationError(
            f"{key} must be set for AF3 execution backend {backend!r}. "
            "Use an explicit path rather than a temporary or implicit default."
        )
    return _expanded_path(str(value))


def _select_execution_backend(
    run_settings: dict, platform_name: str | None = None
) -> str:
    """Resolves auto to Singularity on Linux and native MPS on macOS."""
    configured = str(run_settings.get("af3_execution_backend", "auto")).lower()
    current_platform = platform_name or platform.system()
    if configured == "auto":
        return "native_mps" if current_platform == "Darwin" else "singularity"
    if configured not in {"native_mps", "singularity"}:
        raise AF3ConfigurationError(
            "af3_execution_backend must be one of: auto, native_mps, singularity; "
            f"got {configured!r}."
        )
    if configured == "native_mps" and current_platform != "Darwin":
        raise AF3ConfigurationError(
            "af3_execution_backend='native_mps' is supported only on macOS."
        )
    return configured


def _tail_file(path: str, max_chars: int = 8000) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()[-max_chars:]
    except OSError:
        return ""


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, 15)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, 9)
            else:
                process.kill()
        except OSError:
            pass


def _run_logged_subprocess(
    run_cmds: list[str], log_path: str, timeout_seconds: float | None = None
) -> None:
    """Runs a subprocess with merged logging, timeout cleanup, and exit propagation."""
    run_cmds = [os.fspath(command) for command in run_cmds]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"Running AF3 subprocess: {' '.join(run_cmds)}")
    print(f"AF3 subprocess log: {log_path}")
    with open(log_path, "w", buffering=1) as log_file:
        log_file.write(f"COMMAND: {json.dumps(run_cmds)}\n")
        log_file.flush()
        popen = subprocess.Popen(
            run_cmds,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            return_code = popen.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(popen)
            raise AF3TimeoutError(
                run_cmds, timeout_seconds, log_path=log_path
            ) from exc

    if return_code:
        output = _tail_file(log_path)
        raise AF3ExecutionError(
            return_code,
            run_cmds,
            output=output,
            log_path=log_path,
        )


def _native_af3_command(
    input_path: str, output_dir: str, run_settings: dict
) -> tuple[list[str], float | None]:
    """Builds the stable Germinal-to-native-AF3 subprocess contract."""
    af3_python = _required_setting(run_settings, "af3_python_path", "native_mps")
    if not os.path.isfile(af3_python):
        raise AF3ConfigurationError(
            f"af3_python_path does not point to a file: {af3_python}"
        )

    runner_path = run_settings.get("af3_runner_path")
    if runner_path:
        runner_path = _expanded_path(str(runner_path))
    else:
        runner_path = str(AF3_REPO_ROOT / "scripts" / "run_af3_mps.py")
    if not os.path.isfile(runner_path):
        raise AF3ConfigurationError(
            f"Native AF3 runner not found: {runner_path}"
        )

    run_data_pipeline = bool(run_settings.get("af3_run_data_pipeline", True))
    db_dir = run_settings.get("af3_db_dir")
    db_manifest = run_settings.get("af3_database_manifest")
    if run_data_pipeline and not db_dir and not db_manifest:
        raise AF3ConfigurationError(
            "Native AF3 data-pipeline execution requires af3_db_dir or "
            "af3_database_manifest. Set af3_run_data_pipeline=false only for an "
            "intentional model-only input."
        )

    configured = {
        "af3_expected_revision": run_settings.get(
            "af3_expected_revision", AF3_EXPECTED_REVISION
        ),
        "af3_num_diffusion_samples": int(
            run_settings.get("af3_num_diffusion_samples", 5)
        ),
        "af3_num_recycles": int(run_settings.get("af3_num_recycles", 10)),
        "af3_timeout_seconds": float(
            run_settings.get("af3_timeout_seconds", 3600)
        ),
    }
    if configured["af3_num_diffusion_samples"] < 1:
        raise AF3ConfigurationError("af3_num_diffusion_samples must be >= 1")
    if configured["af3_num_recycles"] < 1:
        raise AF3ConfigurationError("af3_num_recycles must be >= 1")
    if configured["af3_timeout_seconds"] <= 0:
        raise AF3ConfigurationError("af3_timeout_seconds must be > 0")

    command = [
        af3_python,
        runner_path,
        "--af3-source",
        _required_setting(run_settings, "af3_repo_path", "native_mps"),
        "--expected-af3-revision",
        str(configured["af3_expected_revision"]),
        "--model-dir",
        _required_setting(run_settings, "af3_model_dir", "native_mps"),
        "--tools-bin-dir",
        _required_setting(run_settings, "af3_tools_bin_dir", "native_mps"),
        "--tokamax-patch",
        _required_setting(run_settings, "af3_tokamax_patch_path", "native_mps"),
        "--output-dir",
        output_dir,
        "--json-path",
        input_path,
        "--num-diffusion-samples",
        str(configured["af3_num_diffusion_samples"]),
        "--num-recycles",
        str(configured["af3_num_recycles"]),
        "--timeout-seconds",
        str(configured["af3_timeout_seconds"]),
    ]
    if run_data_pipeline:
        command.append("--run-data-pipeline")
    if db_dir:
        command.extend(["--db-dir", _expanded_path(str(db_dir))])
    if db_manifest:
        command.extend(
            ["--database-manifest", _expanded_path(str(db_manifest))]
        )
    return command, configured["af3_timeout_seconds"]


def _singularity_af3_command(
    input_path: str,
    input_dir: str,
    output_dir: str,
    run_settings: dict,
) -> list[str]:
    """Builds the unchanged Linux/Singularity AF3 command."""
    af3_repo_path = _required_setting(run_settings, "af3_repo_path", "singularity")
    weights_path = _required_setting(run_settings, "af3_model_dir", "singularity")
    databases_path = _required_setting(run_settings, "af3_db_dir", "singularity")
    sif_path = _required_setting(run_settings, "af3_sif_path", "singularity")
    return [
        "singularity",
        "exec",
        "--nv",
        "--env",
        "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu",
        "--bind",
        f"{output_dir}:/root/af_output",
        "--bind",
        f"{input_dir}:/root/af_input",
        "--bind",
        f"{weights_path}:/root/models",
        "--bind",
        f"{databases_path}:/root/public_databases",
        "--bind",
        f"{af3_repo_path}:/root/alphafold3",
        sif_path,
        "python",
        "/root/alphafold3/run_alphafold.py",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        f"--output_dir={output_dir}",
        f"--json_path={input_path}",
    ]


def create_input_dict(
    binder_seq, target_seq, binder_chain, target_chains, design_name, seed
):
    """
    Create input JSON data for AlphaFold3 inference.

    Args:
        binder_seq (str): Amino acid sequence of the binder protein.
        target_seq (str): Amino acid sequence of the target protein.
        binder_chain (str): Chain ID for the binder protein.
        target_chains (Union[str, List[str]]): Chain ID(s) for the target protein.
        design_name (str): Name identifier for this design.
        seed (Union[int, List[int]]): Random seed(s) for model inference.

    Returns:
        dict: AF3-compatible input JSON structure.
    """
    if isinstance(target_chains, str):
        target_chains = target_chains.split(',')
    if isinstance(target_seq, str):
        target_sequences = [target_seq]
    else:
        target_sequences = list(target_seq)
    if len(target_sequences) != len(target_chains):
        raise ValueError(
            "target_seq must contain one sequence per target chain: "
            f"{len(target_sequences)} sequences for {len(target_chains)} chains."
        )
        
    if isinstance(seed, int):
        seed = [seed]

    input_json_data = {
        "name": design_name,
        "modelSeeds": seed,
        "dialect": "alphafold3",
        "version": 2,
    }

    sequences = []
    for chain_id in [binder_chain] + target_chains:
        if chain_id != binder_chain:
            i = target_chains.index(chain_id)
            chain_sequence = target_sequences[i]
        else:
            chain_sequence = binder_seq
        sequences.append(
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": chain_sequence,
                }
            }
        )
    input_json_data["sequences"] = sequences
    return input_json_data


def remove_a3m_insertions(a3m_path):
    """
    Remove insertion characters from A3M MSA file for AF3 compatibility.

    AlphaFold3 requires MSA sequences to have uniform length, so we remove
    lowercase insertion characters that indicate gaps in the alignment.

    Args:
        a3m_path (str): Path to the A3M format MSA file to process.
    """
    with open(a3m_path, "r") as a3m_file:
        lines = a3m_file.readlines()
    new_lines = []
    for line in lines:
        line = line.replace("\x00", "")
        if line.startswith("#") or line.startswith(">"):
            new_lines.append(line)
        else:
            new_lines.append("".join(c for c in line if not c.islower()))
    with open(a3m_path, "w") as a3m_file:
        a3m_file.writelines(new_lines)


def generate_local_msa(
    sequence,
    design_name,
    output_dir,
    msa_db_dir,
    use_gpu=False,
    use_gpu_server=False,
    use_metagenomic_db=False,
):
    """
    Generate an unpaired MSA for the given sequence using colabfold_search.

    Args:
        sequence (str): The sequence to generate an MSA for.
        design_name (str): The name of the design. Used to name the output file.
        output_dir (str): The directory to save the output files to.
        msa_db_dir (str): The directory containing the MSA databases.
        use_gpu (bool): Whether to use a GPU for the search.
        use_gpu_server (bool): Whether to use a GPU server for the search.
        use_metagenomic_db (bool): Whether to use the metagenomic database.
    """
    # Start GPU server processes for accelerated MSA search
    if use_gpu_server:
        print("Starting GPU server...")
        gpu_server_dir = os.path.join(msa_db_dir, "colabfold_envdb_202108_db")
        uniref30_db_dir = os.path.join(msa_db_dir, "uniref30_2302_db")
        gpu_server_process = subprocess.Popen(
            [
                "mmseqs",
                "gpuserver",
                gpu_server_dir,
                "--max-seqs",
                "10000",
                "--db-load-mode",
                "0",
                "--prefilter-mode",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("GPU server started at PID", gpu_server_process.pid)
        gpu_server_process.wait()
        uniref30_server_process = subprocess.Popen(
            [
                "mmseqs",
                "gpuserver",
                uniref30_db_dir,
                "--max-seqs",
                "10000",
                "--db-load-mode",
                "0",
                "--prefilter-mode",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Uniref30 server started at PID", uniref30_server_process.pid)
        uniref30_server_process.wait()

    # Create temporary FASTA file for sequence input
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, f"{design_name}.fasta")
        with open(fasta_path, "w") as fasta_file:
            fasta_file.write(f">{design_name}\n{sequence}\n")
        # Create output directory for MSA results
        msa_out_dir = os.path.join(output_dir, "msas")
        os.makedirs(msa_out_dir, exist_ok=True)
        # Build ColabFold search command with appropriate flags
        cmd = ["colabfold_search"]
        if use_gpu:
            cmd += ["--gpu", "1"]
        if use_gpu_server:
            cmd += ["--gpu-server", "1"]
        if not use_metagenomic_db:
            cmd += ["--use-env", "0"]
        cmd += [fasta_path, msa_db_dir, msa_out_dir]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"colabfold_search failed for {design_name}: {e}. Falling back to no MSA."
            )
            return ""
        # Locate generated A3M file and rename it appropriately
        a3m_file = os.path.join(msa_out_dir, f"0.a3m")
        if os.path.exists(a3m_file):
            # Rename MSA file to match design name
            shutil.move(a3m_file, os.path.join(msa_out_dir, f"{design_name}.a3m"))
            a3m_file = os.path.join(msa_out_dir, f"{design_name}.a3m")
            # Process MSA to ensure uniform sequence length for AF3
            remove_a3m_insertions(a3m_file)
            return os.path.relpath(a3m_file, output_dir)
        else:
            print(
                f"colabfold_search failed for {design_name}: MSA not found at {a3m_file}. Falling back to no MSA."
            )
            return ""


def call_generate_colabfold_msa_with_timeout(
    sequence, design_name, output_dir, timeout=120, use_metagenomic_db=False
):
    """
    Generate MSA via ColabFold API with timeout protection.

    Uses ProcessPoolExecutor to enforce a timeout on MSA generation.
    If timeout is exceeded, the worker process is terminated and an empty
    MSA path is returned to allow AF3 to continue without MSA.

    Args:
        sequence (str): Protein sequence for MSA generation.
        design_name (str): Design identifier for output naming.
        output_dir (str): Directory to save MSA output.
        timeout (int): Maximum time in seconds to wait for MSA.
        use_metagenomic_db (bool): Whether to include metagenomic databases.

    Returns:
        str: Relative path to generated MSA file, or empty string if failed.
    """

    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(
            generate_colabfold_msa,
            sequence,
            design_name,
            output_dir,
            use_metagenomic_db,
        )
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            # Cancel and force-shutdown the pool, killing the worker process.
            fut.cancel()
            exe.shutdown(wait=False, cancel_futures=True)
            print(
                f"colabfold_search failed for {design_name}: timed out after {timeout}s. Returning empty MSA."
            )
            return ""


def generate_colabfold_msa(sequence, design_name, output_dir, use_metagenomic_db=False):
    """
    Generate unpaired MSA using ColabFold's remote API.

    This function calls the ColabFold web service to generate a multiple
    sequence alignment for the input protein sequence.

    Args:
        sequence (str): Protein sequence for MSA generation.
        design_name (str): Design identifier for output file naming.
        output_dir (str): Directory to save MSA results.
        use_metagenomic_db (bool): Whether to search metagenomic databases.

    Returns:
        str: Relative path to generated MSA file, or empty string if failed.
    """
    try:
        print(
            f"Running colabfold_search for {design_name} with use_env={use_metagenomic_db}"
        )
        run_mmseqs2(
            sequence,
            os.path.join(output_dir, f"{design_name}"),
            use_env=use_metagenomic_db,
        )
        print(f"colabfold_search finished for {design_name}")
    except Exception as e:
        print(
            f"colabfold_search failed for {design_name}: {e}. Falling back to no MSA."
        )
        return ""

    old_msa_path = os.path.join(output_dir, f"{design_name}_all", "uniref.a3m")
    # Process MSA to ensure uniform sequence length for AF3 compatibility
    remove_a3m_insertions(old_msa_path)
    if os.path.exists(old_msa_path):
        new_msa_path = os.path.join(output_dir, f"msas/{design_name}.a3m")
        os.makedirs(os.path.dirname(new_msa_path), exist_ok=True)
        shutil.copyfile(old_msa_path, new_msa_path)
        shutil.rmtree(os.path.join(output_dir, f"{design_name}_all"))
        return os.path.relpath(new_msa_path, output_dir)
    else:
        print(
            f"colabfold_search failed for {design_name}: MSA not found at {old_msa_path}. Falling back to no MSA."
        )
        return ""


def generate_msas(
    input_json_data: dict,
    msa_db_dir: str,
    output_dir: str,
    binder_chain: str,
    msa_mode: str,
    use_metagenomic_db: bool = False,
    cache_binder_msa: bool = False,
) -> dict:
    """
    Generate Multiple Sequence Alignments (MSAs) for protein chains.

    Creates MSAs for each protein chain in the input data using the specified
    method. MSAs improve structure prediction accuracy by providing evolutionary
    context. The function supports different MSA generation modes:
    - "local": Use local ColabFold search with databases
    - "colabfold": Use ColabFold remote API
    - "target": Generate MSA only for target protein

    When ``cache_binder_msa`` is True, the first binder MSA generated for a run
    is cached to ``msas/binder_cached.a3m`` and reused for subsequent designs by
    rewriting only the query line (line 1), as long as the new binder sequence
    has the same length as the cached query. This works because
    ``generate_local_msa`` / colabfold search strip lowercase insertions, so
    cached rows have uniform length == query length; swapping line 1 keeps
    position k in every row aligned to position k of the query. Length mismatch
    falls through to full regeneration.

    Args:
        input_json_data (dict): AF3 input JSON containing sequence information.
        msa_db_dir (str): Path to local MSA databases (for local mode).
        output_dir (str): Directory to save generated MSA files.
        binder_chain (str): Chain identifier for the binder protein.
        msa_mode (str): MSA generation method.
        use_metagenomic_db (bool): Include metagenomic databases in search.
        cache_binder_msa (bool): Enable binder MSA caching with query-line
            rewrite. Default False.

    Returns:
        dict: Updated input JSON with MSA paths added to each sequence.
    """
    def set_unpaired_msa_path(sequence_info: dict, msa_path: str) -> None:
        if not msa_path:
            return
        sequence_info["protein"]["unpairedMsaPath"] = msa_path
        # AF3 requires a complete custom MSA/template specification. The
        # existing target-MSA mode intentionally uses no paired MSA or
        # templates for this chain.
        sequence_info["protein"]["pairedMsa"] = ""
        sequence_info["protein"]["templates"] = []

    updated_sequences = []
    for seq_idx, sequence_info in enumerate(input_json_data["sequences"]):
        chain = sequence_info["protein"]["id"][0]
        sequence = sequence_info["protein"]["sequence"]
        if chain != binder_chain:
            # Check if target MSA already exists to avoid regeneration
            design_name = f"target_{chain}"
            relative_msa_path = os.path.join(f"msas/{design_name}.a3m")
            full_msa_path = os.path.join(output_dir, relative_msa_path)

            if os.path.exists(full_msa_path):
                set_unpaired_msa_path(sequence_info, relative_msa_path)
                updated_sequences.append(sequence_info)
                continue
        else:
            design_name = input_json_data["name"]
            # Cache-hit branch: reuse cached binder MSA by swapping only the query
            # line. Rows have uniform length == query length (insertions stripped
            # by generate_local_msa / colabfold), so line-1 swap keeps the
            # homolog alignment valid. Length mismatch falls through to regen.
            if cache_binder_msa:
                cache_path = os.path.join(output_dir, "msas", "binder_cached.a3m")
                if os.path.exists(cache_path):
                    with open(cache_path) as _f:
                        cached_lines = _f.readlines()
                    cached_query = cached_lines[1].strip() if len(cached_lines) >= 2 else ""
                    if len(cached_query) == len(sequence):
                        new_lines = list(cached_lines)
                        new_lines[0] = f">{design_name}\n"
                        new_lines[1] = f"{sequence.upper()}\n"
                        relative_msa_path = os.path.join("msas", f"{design_name}_binder.a3m")
                        full_msa_path = os.path.join(output_dir, relative_msa_path)
                        os.makedirs(os.path.dirname(full_msa_path), exist_ok=True)
                        with open(full_msa_path, "w") as _f:
                            _f.writelines(new_lines)
                        set_unpaired_msa_path(sequence_info, relative_msa_path)
                        updated_sequences.append(sequence_info)
                        continue
            # Skip binder MSA generation when mode is target-only (unless caching)
            if msa_mode == "target":
                updated_sequences.append(sequence_info)
                continue

        # Generate MSA using the specified method
        if msa_mode == "local":
            relative_msa_path = generate_local_msa(
                sequence,
                design_name,
                output_dir,
                msa_db_dir,
                use_metagenomic_db=use_metagenomic_db,
            )
        elif msa_mode == "colabfold" or msa_mode == "target":
            relative_msa_path = call_generate_colabfold_msa_with_timeout(
                sequence, design_name, output_dir, use_metagenomic_db=use_metagenomic_db
            )
        else:
            print(f"MSA mode {msa_mode} not recognized. Skipping MSA generation.")

        # Seed the binder MSA cache on first generation. Subsequent designs with
        # a same-length binder will reuse this alignment via the cache-hit
        # branch above instead of regenerating from scratch.
        if cache_binder_msa and chain == binder_chain and relative_msa_path:
            cache_path = os.path.join(output_dir, "msas", "binder_cached.a3m")
            if not os.path.exists(cache_path):
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                shutil.copy(os.path.join(output_dir, relative_msa_path), cache_path)

        set_unpaired_msa_path(sequence_info, relative_msa_path)

        updated_sequences.append(sequence_info)
        if relative_msa_path != "":
            print(
                f"Generated MSA at {relative_msa_path} for {design_name} and sequence {seq_idx}"
            )
        else:
            print(f"No MSA generated for {design_name} and sequence {seq_idx}")
    input_json_data["sequences"] = updated_sequences
    return input_json_data


def _af3_output_paths(
    output_dir: str, design_name: str, select_mode: str = "best"
) -> tuple[str, str, str]:
    """Locates AF3 structure and confidence outputs and fails closed if incomplete."""
    design_name = design_name.lower()
    af3_results_folder = os.path.join(output_dir, design_name)

    if select_mode == "worst":
        ranking_csv = os.path.join(
            af3_results_folder, f"{design_name}_ranking_scores.csv"
        )
        if not os.path.isfile(ranking_csv):
            raise FileNotFoundError(
                f"AF3 ranking CSV was not produced: {ranking_csv}"
            )
        ranking_df = pd.read_csv(ranking_csv)
        if ranking_df.empty or "ranking_score" not in ranking_df:
            raise ValueError(
                f"AF3 ranking CSV has no usable ranking_score rows: {ranking_csv}"
            )
        worst_row = ranking_df.loc[ranking_df["ranking_score"].idxmin()]
        seed = int(worst_row["seed"])
        sample = int(worst_row["sample"])
        sample_subfolder = f"seed-{seed}_sample-{sample}"
        sample_prefix = f"{design_name}_{sample_subfolder}"
        sample_folder = os.path.join(af3_results_folder, sample_subfolder)
        af3_structure = os.path.join(
            sample_folder, f"{sample_prefix}_model.cif"
        )
        summary_confidences = os.path.join(
            sample_folder, f"{sample_prefix}_summary_confidences.json"
        )
        full_confidences = os.path.join(
            sample_folder, f"{sample_prefix}_confidences.json"
        )
        print(
            f"Selected worst sample for {design_name}: {sample_subfolder} "
            f"(ranking_score={worst_row['ranking_score']:.4f})"
        )
    elif select_mode == "best":
        af3_structure = os.path.join(
            af3_results_folder, f"{design_name}_model.cif"
        )
        summary_confidences = os.path.join(
            af3_results_folder, f"{design_name}_summary_confidences.json"
        )
        full_confidences = os.path.join(
            af3_results_folder, f"{design_name}_confidences.json"
        )
    else:
        raise ValueError(
            f"Unsupported AF3 select_mode={select_mode!r}; expected 'best' or 'worst'."
        )

    missing = [
        path
        for path in (af3_structure, summary_confidences, full_confidences)
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "AF3 completed without the expected output artifacts. Missing: "
            + ", ".join(missing)
        )
    return af3_structure, summary_confidences, full_confidences


def extract_structure_and_scores(output_dir, design_name, binder_chain, select_mode="best"):
    """
    Extract predicted structure and confidence scores from AF3 output.

    Processes AF3 results by converting the output CIF file to PDB format,
    extracting confidence metrics from JSON files, and cleaning up temporary
    files to save disk space.

    Args:
        output_dir (str): Directory containing AF3 output folder.
        design_name (str): Name identifier for the design.
        binder_chain (str): Chain ID for the binder protein.
        select_mode (str): Which sample to select from the AF3 output.
            - "best": Use the top-ranked sample (AF3 default, top-level files).
            - "worst": Use the lowest-ranked sample based on ranking_score
              from the ranking_scores.csv file.
            Defaults to "best".

    Returns:
        tuple: (pdb_path, scores_dict, ipsae) where:
            - pdb_path (str): Path to converted PDB structure file
            - scores_dict (dict): Confidence metrics including pLDDT, PAE, pTM, iPTM
            - ipsae (dict or None): ipSAE, pDockQ2, and LIS scores if available
    """

    # AF3 lowercases the design name in output folder and file names
    design_name = design_name.lower()

    af3_results_folder = os.path.join(output_dir, design_name)
    af3_structure, summary_confidences, full_confidences = _af3_output_paths(
        output_dir, design_name, select_mode=select_mode
    )

    # Convert mmCIF structure file to PDB format for compatibility
    pdb_path = os.path.join(output_dir, f"{design_name}_af3.pdb")
    parser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("structure", af3_structure)
    io.set_structure(structure)
    io.save(pdb_path)
    af3_scores = {}
    with open(summary_confidences, "r") as f:
        summary_metrics = json.load(f)
    with open(full_confidences, "r") as f:
        full_metrics = json.load(f)
    #plddt
    binder_plddt = [x for x, c in zip(full_metrics["atom_plddts"], full_metrics["atom_chain_ids"]) if c == binder_chain]
    af3_scores["plddt_binder"] = np.mean(binder_plddt) / 100 if binder_plddt else 0
    af3_scores["plddt"] = np.mean(full_metrics["atom_plddts"]) / 100
    #pae
    pae_matrix = np.array(full_metrics["pae"])
    af3_scores["pae_matrix"] = pae_matrix
    af3_scores["pae"] = np.mean(pae_matrix)
    af3_scores["chain_pair_pae_min"] = np.array(summary_metrics['chain_pair_pae_min'])
    binder_mask = np.array(full_metrics['token_chain_ids']) == binder_chain
    pae_chain = pae_matrix[np.ix_(binder_mask, binder_mask)]  # double mask: rows AND columns
    af3_scores["binder_pae"] = np.mean(pae_chain)
    #ptm
    af3_scores["ptm"] = [summary_metrics["ptm"]]
    af3_scores["iptm"] = [summary_metrics["iptm"]]
    af3_scores["chain_iptm"] = np.array(summary_metrics["chain_iptm"])
    af3_scores['chain_ptm'] = np.array(summary_metrics["chain_ptm"])
    #agg score
    af3_scores["aggregate_score"] = [summary_metrics["ranking_score"]]
    
    try:
        ipsae_result = subprocess.run(
            ["ipsae", full_confidences, af3_structure, "10", "10"],
            capture_output=True,
            text=True
        )
        ipsae_out_file = f"{os.path.dirname(os.path.abspath(af3_structure))}/{os.path.basename(af3_structure).replace('.cif','_10_10.txt')}"
        df = pd.read_csv(ipsae_out_file, sep=r"\s+")
        ipsae_score = df['ipSAE'][len(df['ipSAE'])-1]
        pdockq2_score = df['pDockQ2'][len(df['pDockQ2'])-1]
        lis_score = df['LIS'][len(df['LIS'])-1]
        ipsae = {'ipsae':ipsae_score, 'pdockq2':pdockq2_score, 'LIS':lis_score}
    except:
        print(f'ipSAE calculation failed!')
        ipsae = None

    # Clean up temporary AF3 job folder to save disk space
    shutil.rmtree(af3_results_folder)

    return pdb_path, af3_scores, ipsae


def _run_af3(
    input_json: dict,
    output_dir: str,
    binder_chain: str,
    msa_mode: str,
    run_settings: dict,
    select_mode: str = "best",
) -> tuple:
    """
    Execute AlphaFold3 through the platform-appropriate subprocess boundary.

    macOS uses the validated native MPS runner in the dedicated AF3 environment.
    Other platforms retain the existing Linux/Singularity execution path.

    Args:
        input_json (dict): AF3-compatible input JSON with sequence information.
        output_dir (str): Directory to save prediction outputs.
        binder_chain (str): Chain identifier for the binder protein.
        msa_mode (str): MSA generation mode ("none", "local", "colabfold", "target").
        run_settings (dict): Configuration containing all AF3 paths and settings.
        select_mode (str): Which sample to extract ("best" or "worst").

    Returns:
        tuple: (pdb_path, scores_dict, ipsae) where:
            - pdb_path (str): Path to predicted structure in PDB format
            - scores_dict (dict): Confidence metrics and scores
            - ipsae (dict or None): ipSAE, pDockQ2, and LIS scores if available
    """
    # Verify output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created at {output_dir}.")

    # Process input. if a path, load the dict, and get the dir name. otherwise, use a temp dir.
    input_dir = os.path.join(output_dir, "af3_inputs")
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"{input_json['name']}.json")
    # Generate MSAs if requested. pass input_dir as the output_dir to save msas there
    if msa_mode in ["local", "colabfold", "target"]:
        input_json = generate_msas(
            input_json,
            run_settings["msa_db_dir"],
            input_dir,
            binder_chain,
            msa_mode,
            use_metagenomic_db=run_settings["use_metagenomic_db"],
            cache_binder_msa=run_settings.get("cache_binder_msa", False),
        )

    # Write the exact AF3 input consumed by either execution backend.
    with open(input_path, "w") as f:
        json.dump(input_json, f, indent=2)

    execution_backend = _select_execution_backend(run_settings)
    log_path = os.path.join(
        input_dir, f"{input_json['name']}_af3_subprocess.log"
    )
    if execution_backend == "native_mps":
        run_cmds, declared_timeout = _native_af3_command(
            input_path, output_dir, run_settings
        )
        # The native runner enforces its own timeout and kills the complete AF3
        # process group. The parent allows a small grace period for cleanup.
        parent_timeout = declared_timeout + 30
    else:
        run_cmds = _singularity_af3_command(
            input_path, input_dir, output_dir, run_settings
        )
        parent_timeout = run_settings.get("af3_singularity_timeout_seconds")
        if parent_timeout is not None:
            parent_timeout = float(parent_timeout)
            if parent_timeout <= 0:
                raise AF3ConfigurationError(
                    "af3_singularity_timeout_seconds must be > 0 when set."
                )

    _run_logged_subprocess(
        run_cmds, log_path=log_path, timeout_seconds=parent_timeout
    )

    pdb_path, scores, ipsae = extract_structure_and_scores(
        output_dir, input_json["name"], binder_chain, select_mode=select_mode
    )

    return pdb_path, scores, ipsae


def run_af3(
    binder_seq: str,
    target_seq: Union[str, List[str]],
    target_chains: Union[List[str], str],
    output_dir: str,
    design_name: str,
    seed: Union[int, List[int]],
    run_settings: dict,
    binder_chain: str = "B",
    msa_mode: str = "none",
    select_mode: str = "best",
):
    """
    Run AlphaFold3 structure prediction for antibody-target complex.

    High-level interface for AF3 structure prediction. Creates input JSON,
    generates MSAs if requested, runs AF3 inference, and returns the predicted
    structure with confidence scores.

    Args:
        binder_seq (str): Amino acid sequence of the antibody binder.
        target_seq (str): Amino acid sequence of the target protein.
        target_chains (Union[List[str], str]): Chain identifier(s) for target protein.
        output_dir (str): Directory to save prediction outputs.
        design_name (str): Unique identifier for this design.
        seed (Union[int, List[int]]): Random seed(s) for reproducible predictions.
        run_settings (dict): Configuration containing AF3 paths and MSA settings:
            - af3_execution_backend: "auto", "native_mps", or "singularity"
            - af3_repo_path: Path to the pinned AlphaFold3 repository
            - af3_sif_path: Path to AF3 Singularity image (Linux only)
            - af3_model_dir: Path to AF3 model weights
            - af3_db_dir: Path to AF3 public databases
            - af3_python_path: Dedicated AF3 interpreter (macOS only)
            - af3_tools_bin_dir: HMMER tools directory (macOS only)
            - af3_tokamax_patch_path: Reversible MPS precision patch
            - af3_database_manifest: Optional explicit database path map
            - af3_run_data_pipeline: Whether AF3 database search runs
            - af3_num_diffusion_samples: AF3 diffusion sample count
            - af3_num_recycles: AF3 recycle count
            - af3_timeout_seconds: Native AF3 subprocess timeout
            - msa_db_dir: Path to ColabFold MSA databases
            - use_metagenomic_db: Whether to use metagenomic databases
        binder_chain (str, optional): Chain ID for binder protein. Defaults to 'B'.
        msa_mode (str, optional): MSA generation method:
            - 'none': No MSA generation
            - 'local': Use local ColabFold databases
            - 'colabfold': Use ColabFold remote API
            - 'target': Generate MSA only for target protein
            Defaults to 'none'.
        select_mode (str, optional): Which sample to extract from AF3 output.
            - 'best': Use the top-ranked sample (AF3 default).
            - 'worst': Use the lowest-ranked sample by ranking_score.
            Defaults to 'best'.

    Returns:
        tuple: (pdb_path, scores_dict, ipsae) where:
            - pdb_path (str): Path to predicted complex structure (PDB format)
            - scores_dict (dict): Confidence metrics including pLDDT, PAE, pTM, iPTM
            - ipsae (dict or None): ipSAE, pDockQ2, and LIS scores if available
    """
    # Gate: cache_binder_msa relies on a real binder MSA being generated first
    # (the first binder seeds the cache via shutil.copy, subsequent ones rewrite
    # only the query line). With msa_mode="target" no binder MSA is ever
    # generated, so the cache cannot seed and the flag silently does nothing.
    # Local mode would work in theory but is untested. Require colabfold.
    if run_settings.get("cache_binder_msa", False) and msa_mode != "colabfold":
        raise ValueError(
            f"cache_binder_msa=True requires msa_mode='colabfold'; got "
            f"msa_mode={msa_mode!r}. In any other mode the binder MSA is "
            f"either skipped (target) or untested (local/none), so the cache "
            f"cannot seed and the flag silently does nothing. Set "
            f"cache_binder_msa=false or switch msa_mode to 'colabfold'."
        )

    input_json_data = create_input_dict(
        binder_seq, target_seq, binder_chain, target_chains, design_name, seed
    )
    return _run_af3(
        input_json_data,
        output_dir,
        binder_chain=binder_chain,
        msa_mode=msa_mode,
        run_settings=run_settings,
        select_mode=select_mode,
    )


def main(
    input_json: str,
    output_dir: str,
    msa_db_dir: str,
    binder_chain: str,
    msa_mode: str,
    af3_repo_path: str,
    af3_sif_path: str,
    af3_model_dir: str,
    af3_db_dir: str,
    use_metagenomic_db: bool,
    af3_execution_backend: str = "auto",
    af3_expected_revision: str = AF3_EXPECTED_REVISION,
    af3_python_path: str | None = None,
    af3_tools_bin_dir: str | None = None,
    af3_runner_path: str | None = None,
    af3_tokamax_patch_path: str | None = None,
    af3_database_manifest: str | None = None,
    af3_run_data_pipeline: bool = True,
    af3_num_diffusion_samples: int = 5,
    af3_num_recycles: int = 10,
    af3_timeout_seconds: float = 3600,
):
    print("Running AF3...")

    results = pd.DataFrame(columns=["name", "plddt", "iptm"])
    with open(input_json) as f:
        input_json_data = json.load(f)
        if isinstance(input_json_data, list):
            print(
                f"Detected list input with {len(input_json_data)} items. Running AF3 for each."
            )
            for i, input_json_dict in enumerate(input_json_data):
                run_settings = {
                    "msa_db_dir": msa_db_dir,
                    "af3_repo_path": af3_repo_path,
                    "af3_sif_path": af3_sif_path,
                    "af3_model_dir": af3_model_dir,
                    "af3_db_dir": af3_db_dir,
                    "af3_execution_backend": af3_execution_backend,
                    "af3_expected_revision": af3_expected_revision,
                    "af3_python_path": af3_python_path,
                    "af3_tools_bin_dir": af3_tools_bin_dir,
                    "af3_runner_path": af3_runner_path,
                    "af3_tokamax_patch_path": af3_tokamax_patch_path,
                    "af3_database_manifest": af3_database_manifest,
                    "af3_run_data_pipeline": af3_run_data_pipeline,
                    "af3_num_diffusion_samples": af3_num_diffusion_samples,
                    "af3_num_recycles": af3_num_recycles,
                    "af3_timeout_seconds": af3_timeout_seconds,
                    "use_metagenomic_db": use_metagenomic_db,
                }
                pdb_path, scores, _ = _run_af3(
                    input_json_dict,
                    output_dir,
                    binder_chain=binder_chain,
                    msa_mode=msa_mode,
                    run_settings=run_settings,
                )
                results = results.append(
                    {
                        "name": input_json_dict["name"],
                        "plddt": scores["plddt"],
                        "iptm": scores["iptm"],
                    },
                    ignore_index=True,
                )
                # save pae matrix
                pae_matrix_path = os.path.join(
                    output_dir, f"{input_json_dict['name']}_pae_matrix.npy"
                )
                np.save(pae_matrix_path, scores["pae_matrix"])

                print(
                    f"Folded {i + 1}/{len(input_json_data)} AF3 structures: design {input_json_dict['name']}, iptm {scores['iptm']}, plddt {scores['plddt']}"
                )
        else:
            run_settings = {
                "msa_db_dir": msa_db_dir,
                "af3_repo_path": af3_repo_path,
                "af3_sif_path": af3_sif_path,
                "af3_model_dir": af3_model_dir,
                "af3_db_dir": af3_db_dir,
                "af3_execution_backend": af3_execution_backend,
                "af3_expected_revision": af3_expected_revision,
                "af3_python_path": af3_python_path,
                "af3_tools_bin_dir": af3_tools_bin_dir,
                "af3_runner_path": af3_runner_path,
                "af3_tokamax_patch_path": af3_tokamax_patch_path,
                "af3_database_manifest": af3_database_manifest,
                "af3_run_data_pipeline": af3_run_data_pipeline,
                "af3_num_diffusion_samples": af3_num_diffusion_samples,
                "af3_num_recycles": af3_num_recycles,
                "af3_timeout_seconds": af3_timeout_seconds,
                "use_metagenomic_db": use_metagenomic_db,
            }
            pdb_path, scores, _ = _run_af3(
                input_json_data,
                output_dir,
                binder_chain=binder_chain,
                msa_mode=msa_mode,
                run_settings=run_settings,
            )
            results = results.append(
                {
                    "name": input_json_data["name"],
                    "plddt": scores["plddt"],
                    "iptm": scores["iptm"],
                },
                ignore_index=True,
            )
            # save pae matrix
            pae_matrix_path = os.path.join(
                output_dir, f"{input_json_data['name']}_pae_matrix.npy"
            )
            np.save(pae_matrix_path, scores["pae_matrix"])

    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        "-i",
        required=True,
        help="Path to input json file. Must be absolute",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Output directory for AF3 outputs."
    )
    parser.add_argument(
        "--msa_db_dir",
        "-d",
        required=False,
        help="Path to MSA database directory for ColabFold.",
    )
    parser.add_argument(
        "--binder_chain",
        "-b",
        required=False,
        default="B",
        help="Chain ID of the binder chain.",
    )
    parser.add_argument(
        "--msa_mode",
        "-m",
        required=False,
        default="colabfold",
        help="MSA mode. Can be either 'local' or 'colabfold' or 'target' for target only (colabfold default).",
    )
    parser.add_argument(
        "--af3_repo_path", required=False, help="Path to local AlphaFold3 repo to bind."
    )
    parser.add_argument(
        "--af3_execution_backend",
        default="auto",
        choices=["auto", "native_mps", "singularity"],
        help="AF3 execution backend. auto uses native MPS on macOS.",
    )
    parser.add_argument(
        "--af3_expected_revision",
        default=AF3_EXPECTED_REVISION,
        help="Required AF3 Git commit for native MPS execution.",
    )
    parser.add_argument(
        "--af3_sif_path",
        required=False,
        help="Path to the AlphaFold3 Singularity image (.sif).",
    )
    parser.add_argument(
        "--af3_model_dir", required=False, help="Path to AF3 model weights directory."
    )
    parser.add_argument(
        "--af3_db_dir", required=False, help="Path to AF3 public databases directory."
    )
    parser.add_argument(
        "--af3_python_path",
        required=False,
        help="Interpreter in the dedicated AF3 environment (native macOS).",
    )
    parser.add_argument(
        "--af3_tools_bin_dir",
        required=False,
        help="Directory containing native HMMER tools (native macOS).",
    )
    parser.add_argument(
        "--af3_runner_path",
        required=False,
        help="Path to scripts/run_af3_mps.py.",
    )
    parser.add_argument(
        "--af3_tokamax_patch_path",
        required=False,
        help="Path to the reversible Tokamax MPS precision patch.",
    )
    parser.add_argument(
        "--af3_database_manifest",
        required=False,
        help="Optional JSON mapping of all nine explicit AF3 database paths.",
    )
    parser.add_argument(
        "--af3_run_data_pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether native AF3 runs MSA/template database search.",
    )
    parser.add_argument(
        "--af3_num_diffusion_samples", type=int, default=5
    )
    parser.add_argument("--af3_num_recycles", type=int, default=10)
    parser.add_argument(
        "--af3_timeout_seconds", type=float, default=3600
    )
    parser.add_argument(
        "--use_metagenomic_db",
        action="store_true",
        help="Use metagenomic database for MSA generation.",
    )
    args = parser.parse_args()

    main(
        input_json=args.input_json,
        output_dir=args.output_dir,
        msa_db_dir=args.msa_db_dir,
        binder_chain=args.binder_chain,
        msa_mode=args.msa_mode,
        af3_repo_path=args.af3_repo_path,
        af3_sif_path=args.af3_sif_path,
        af3_model_dir=args.af3_model_dir,
        af3_db_dir=args.af3_db_dir,
        use_metagenomic_db=args.use_metagenomic_db,
        af3_execution_backend=args.af3_execution_backend,
        af3_expected_revision=args.af3_expected_revision,
        af3_python_path=args.af3_python_path,
        af3_tools_bin_dir=args.af3_tools_bin_dir,
        af3_runner_path=args.af3_runner_path,
        af3_tokamax_patch_path=args.af3_tokamax_patch_path,
        af3_database_manifest=args.af3_database_manifest,
        af3_run_data_pipeline=args.af3_run_data_pipeline,
        af3_num_diffusion_samples=args.af3_num_diffusion_samples,
        af3_num_recycles=args.af3_num_recycles,
        af3_timeout_seconds=args.af3_timeout_seconds,
    )
