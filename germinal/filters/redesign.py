"""
AbMPNN redesign utilities for Germinal.

Attribution:
- The AbMPNN model weights (arXiv:2310.19513) were presented at the 2023 ICML Workshop on Computational Biology.
- Model weights and CSV files with the train, test, and validation splits across the SAbDab and ImmuneBuilder datasets are provided by the authors.
- AbMPNN is based on ProteinMPNN and can be run using the corresponding code: https://github.com/dauparas/ProteinMPNN.
"""

import os
import shutil
import tempfile
import multiprocessing as mp
import pickle
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from germinal.utils.utils import hotspot_residues, clear_memory, get_sequence_from_pdb
from colabdesign.mpnn import mk_mpnn_model

mp.set_start_method("spawn", force=True)


def abmpnn_design(
    trajectory_pdb: str,
    trajectory_interface_residues: str,
    run_settings: Dict[str, Any],
    target_chain: str = "A",
    binder_chain: str = "B",
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Generate redesigned sequences using AbMPNN for a given PDB structure.

    Args:
        trajectory_pdb: Path to the input PDB file.
        target_chain: Target chain identifier (e.g., 'A').
        binder_chain: Binder chain identifier (e.g., 'B').
        trajectory_interface_residues: Comma-separated string of interface residue indices.
        run_settings: Dictionary of settings.

    Returns:
        Dictionary containing redesigned sequence and associated scores.
    """
    abmpnn_model = mk_mpnn_model(
        backbone_noise=run_settings["backbone_noise"],
        model_name=run_settings["model_path"],
        weights=run_settings["mpnn_weights"],
        seed=seed,
    )

    # Determine which residues to fix during redesign
    design_chains = f"{target_chain},{binder_chain}"
    if run_settings.get("mpnn_fix_interface", False):
        fixed_positions = ",".join(target_chain.split(",") + [str(trajectory_interface_residues)]).rstrip(",")
    else:
        fixed_positions = target_chain

    # Prepare AbMPNN model inputs
    abmpnn_model.prep_inputs(
        pdb_filename=trajectory_pdb,
        chain=design_chains,
        fix_pos=fixed_positions,
        rm_aa=run_settings["omit_AAs"],
    )
    # Sample redesigned sequences
    abmpnn_sequences = abmpnn_model.sample(
        temperature=run_settings["sampling_temp"], num=1, batch=run_settings["num_seqs"]
    )

    # Clean up memory
    del abmpnn_model
    clear_memory()

    return abmpnn_sequences


def abmpnn_worker(
    trajectory_pdb: str,
    target_chain: str,
    binder_chain: str,
    trajectory_interface_residues: str,
    run_settings: Dict[str, Any],
    output_path: str,
    seed: int | None = None,
) -> None:
    """
    Worker function for AbMPNN sequence generation in multiprocessing.

    On exception, dumps the full traceback to ``{output_path}.err`` so the
    parent can read it and raise a loud error. Without this, child failures
    only surface as a missing/empty pickle with no diagnostics.

    Args:
        trajectory_pdb: Path to the trajectory PDB file
        target_chain: Target chain identifier
        binder_chain: Binder chain identifier
        trajectory_interface_residues: Interface residues to fix
        run_settings: Dictionary containing AbMPNN settings
        output_path: Path to save the output pickle file
    """
    import traceback
    try:
        result = abmpnn_design(
            trajectory_pdb,
            trajectory_interface_residues,
            run_settings,
            target_chain = target_chain,
            binder_chain = binder_chain,
            seed=seed,
        )
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        err_path = f"{output_path}.err"
        try:
            with open(err_path, "w") as fh:
                fh.write(
                    f"abmpnn_worker failed for trajectory_pdb={trajectory_pdb}\n\n"
                )
                fh.write(traceback.format_exc())
        except Exception:
            pass
        raise


def _abmpnn_output_is_finite(output: Dict[str, Any]) -> bool:
    """Validate the raw model tensors before candidate selection."""
    import numpy as np

    if not output or "seq" not in output or "score" not in output:
        return False
    logits = output.get("logits")
    if logits is None:
        return False
    return bool(
        np.isfinite(np.asarray(logits)).all()
        and np.isfinite(np.asarray(output["score"])).all()
    )


def get_abmpnn_sequences(
    trajectory_pdb_af: str,
    run_settings: Dict[str, Any],
    cdr_positions: List[int],
    atom_distance_cutoff: float = 3.0,
    target_chain: str = "A",
    binder_chain: str = "B",
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate AbMPNN redesigned sequences for a given trajectory.

    Args:
        trajectory_pdb_af: Path to the trajectory PDB file from AF2/ColabDesign
        target_chain: Target chain identifier (e.g., 'A')
        binder_chain: Binder chain identifier (e.g., 'B')
        run_settings: Dictionary containing run settings including:
            - max_mpnn_sequences: Maximum number of MPNN sequences to return
            - cdr_positions: CDR positions to redesign
        atom_distance_cutoff: Distance cutoff for interface residue detection

    Returns:
        List of dictionaries containing MPNN sequences, each with:
            - seq: The redesigned sequence
            - score: MPNN score for the sequence
            - seqid: Sequence identity to original
    """
    chains_pdb = get_sequence_from_pdb(trajectory_pdb_af)
    if len(chains_pdb) < 3:  # Using PDB from AFM which has only 2 chains
        binder_chain = "B"
    length = len(chains_pdb[binder_chain])
    # Always fix framework (non-CDR) positions
    residues_to_fix = [
        f"{binder_chain}{pos+1}"
        for pos in range(0, length)
        if pos not in run_settings["cdr_positions"]
    ]

    if atom_distance_cutoff > 0.0:
        # Also fix interface CDR residues (preserve binding contacts)
        trajectory_interface_residues = hotspot_residues(
            trajectory_pdb_af,
            binder_chain,
            target_chain=target_chain,
            atom_distance_cutoff=atom_distance_cutoff,
        )
        interface_residues_pdb_ids = [
            f"{binder_chain}{pdb_res_num}"
            for pdb_res_num in trajectory_interface_residues.keys()
        ]
        residues_to_fix = set(residues_to_fix + interface_residues_pdb_ids)
    else:
        residues_to_fix = set(residues_to_fix)

    residues_to_fix = ",".join(residues_to_fix)

    # Run MPNN in a dedicated interpreter. A multiprocessing child still
    # re-imports run_germinal.py and can inherit an AF2-contaminated MPS
    # execution state. A dedicated process made the intermittent all-NaN logits
    # much rarer but did not eliminate them, so invalid numerical output is
    # rejected and the same seeded computation is repeated from a fresh process.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as settings_file:
        json.dump(run_settings, settings_file)
        settings_path = settings_file.name

    repo_root = Path(__file__).resolve().parents[2]
    worker_script = repo_root / "scripts" / "run_abmpnn_worker.py"
    configured_python = run_settings.get("abmpnn_python_path")
    if configured_python:
        worker_python = Path(configured_python).expanduser().absolute()
        if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
            raise RuntimeError(
                "Configured AbMPNN interpreter is not executable: "
                f"{worker_python}"
            )
        expected_prefix = worker_python.parent.parent.resolve()
    else:
        worker_python = Path(sys.executable)
        expected_prefix = None
    expected_jax_mps_version = run_settings.get(
        "abmpnn_expected_jax_mps_version"
    )
    command = [
        str(worker_python),
        "-u",
        str(worker_script),
        "--pdb",
        str(Path(trajectory_pdb_af).resolve()),
        "--settings-json",
        settings_path,
        "--fixed-positions",
        residues_to_fix,
        "--target-chain",
        target_chain,
        "--binder-chain",
        binder_chain,
        "--seed",
        str(seed),
    ]
    if expected_jax_mps_version:
        command.extend(
            [
                "--expected-jax-mps-version",
                str(expected_jax_mps_version),
            ]
        )
    if expected_prefix is not None:
        command.extend(["--expected-prefix", str(expected_prefix)])

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not pythonpath
        else str(repo_root) + os.pathsep + pythonpath
    )
    env.pop("PYTHONHOME", None)
    if expected_prefix is not None:
        env.pop("CONDA_PREFIX", None)
        env["VIRTUAL_ENV"] = str(expected_prefix)
        env["PATH"] = (
            str(expected_prefix / "bin")
            + os.pathsep
            + env.get("PATH", "")
        )
    preserve_worker_logs = bool(
        run_settings.get("abmpnn_preserve_worker_logs", False)
    )
    worker_artifact_root = None
    if preserve_worker_logs:
        configured_artifact_root = run_settings.get(
            "abmpnn_worker_log_dir"
        )
        if configured_artifact_root:
            worker_artifact_root = Path(configured_artifact_root).expanduser()
        else:
            worker_artifact_root = (
                Path(trajectory_pdb_af).resolve().parent
                / "abmpnn_workers"
                / Path(trajectory_pdb_af).stem
            )
        worker_artifact_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            settings_path,
            worker_artifact_root / "run_settings.json",
        )
    max_trials = int(os.environ.get("GERMINAL_ABMPNN_MAX_TRIALS", "5"))
    if max_trials < 1:
        raise ValueError("GERMINAL_ABMPNN_MAX_TRIALS must be >= 1")
    abmpnn_trajectories = None
    try:
        for trial in range(1, max_trials + 1):
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                output_path = tf.name
            if worker_artifact_root is None:
                worker_log_path = f"{output_path}.log"
                identity_path = f"{output_path}.runtime.json"
            else:
                worker_log_path = str(
                    worker_artifact_root / f"trial_{trial:02d}.log"
                )
                identity_path = str(
                    worker_artifact_root
                    / f"trial_{trial:02d}.runtime.json"
                )
            trial_command = [
                *command,
                "--output-pickle",
                output_path,
                "--runtime-identity",
                identity_path,
            ]
            with open(worker_log_path, "w") as worker_log:
                proc = subprocess.run(
                    trial_command,
                    env=env,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            if proc.returncode != 0:
                traceback_text = ""
                if os.path.exists(worker_log_path):
                    with open(worker_log_path) as fh:
                        traceback_text = fh.read()[-8000:]
                if os.path.exists(output_path):
                    os.unlink(output_path)
                msg = (
                    f"\n{'=' * 78}\n"
                    f"[ABMPNN ERROR] worker process exited non-zero "
                    f"(exitcode={proc.returncode}) for "
                    f"trajectory_pdb={trajectory_pdb_af}\n"
                    f"{'=' * 78}\n"
                    f"{traceback_text}"
                    f"{'=' * 78}\n"
                )
                print(msg, flush=True)
                raise RuntimeError(
                    f"abmpnn_worker failed (exitcode={proc.returncode}); "
                    f"see traceback above."
                )

            if expected_jax_mps_version or expected_prefix is not None:
                if not os.path.isfile(identity_path):
                    raise RuntimeError(
                        "AbMPNN worker did not emit its runtime identity "
                        f"at {identity_path}"
                    )
                with open(identity_path) as handle:
                    identity = json.load(handle)
                if (
                    expected_jax_mps_version
                    and identity.get("jax_mps_version")
                    != expected_jax_mps_version
                ):
                    raise RuntimeError(
                        "AbMPNN worker imported the wrong jax-mps version: "
                        f"expected {expected_jax_mps_version}, "
                        f"got {identity.get('jax_mps_version')}"
                    )
                if (
                    expected_prefix is not None
                    and identity.get("sys_prefix")
                    != str(expected_prefix)
                ):
                    raise RuntimeError(
                        "AbMPNN worker imported from the wrong environment: "
                        f"expected {expected_prefix}, "
                        f"got {identity.get('sys_prefix')}"
                    )
                if identity.get("jax_default_backend") != "mps":
                    raise RuntimeError(
                        "AbMPNN worker did not select MPS: "
                        f"{identity.get('jax_default_backend')}"
                    )

            try:
                with open(output_path, "rb") as f:
                    candidate = pickle.load(f)
                os.unlink(output_path)
            except Exception as exc:
                if os.path.exists(output_path):
                    os.unlink(output_path)
                raise RuntimeError(
                    "abmpnn_worker exited 0 but produced unreadable pickle "
                    f"at {output_path}: {exc}"
                ) from exc

            if _abmpnn_output_is_finite(candidate):
                abmpnn_trajectories = candidate
                if not preserve_worker_logs:
                    os.unlink(worker_log_path)
                    if os.path.exists(identity_path):
                        os.unlink(identity_path)
                break

            print(
                "AbMPNN produced non-finite logits or scores on Apple MPS; "
                f"discarding trial {trial}/{max_trials} and retrying with the "
                "same seed.",
                flush=True,
            )
            if not preserve_worker_logs:
                os.unlink(worker_log_path)
                if os.path.exists(identity_path):
                    os.unlink(identity_path)
        else:
            raise RuntimeError(
                "AbMPNN produced non-finite logits or scores after "
                f"{max_trials} fresh-process trials; refusing to emit "
                "unvalidated sequences."
            )
    finally:
        if os.path.exists(settings_path):
            os.unlink(settings_path)

    # Process and deduplicate MPNN sequences
    if not abmpnn_trajectories or "seq" not in abmpnn_trajectories:
        print("No MPNN sequences generated")
        return []

    # Create unique sequences dictionary and sort by score
    unique_sequences = {}
    for n in range(len(abmpnn_trajectories["seq"])):
        seq = abmpnn_trajectories["seq"][n][-length:]  # Take only binder sequence
        if seq not in unique_sequences:
            unique_sequences[seq] = {
                "seq": seq,
                "score": abmpnn_trajectories["score"][n],
                "seqid": abmpnn_trajectories["seqid"][n],
            }

    # Sort by AbMPNN score (lower is better) and limit to max sequences
    abmpnn_sequences = sorted(unique_sequences.values(), key=lambda x: x["score"])
    max_sequences = run_settings.get("max_mpnn_sequences", 4)
    abmpnn_sequences = abmpnn_sequences[:max_sequences]

    print(f"Generated {len(abmpnn_sequences)} unique AbMPNN sequences")
    for i, seq_data in enumerate(abmpnn_sequences):
        print(
            f"  Sequence {i + 1}: score={seq_data['score']:.3f}, seqid={seq_data['seqid']:.3f}"
        )

    return abmpnn_sequences


def run_abmpnn_redesign_pipeline(
    trajectory_pdb_af: str,
    run_settings: Dict[str, Any],
    atom_distance_cutoff: float = 3.0,
    seed: int | None = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Complete AbMPNN redesign pipeline for a trajectory.

    Note: target_chain / binder_chain are NOT parameters here — they were
    historically accepted but never forwarded to get_abmpnn_sequences (which
    uses its own defaults of A/B). Removed from the signature to avoid the
    misleading impression that the wrapper honors them.

    Args:
        trajectory_pdb_af: Path to the trajectory PDB file from AF2/ColabDesign
        run_settings: Dictionary containing run settings
        atom_distance_cutoff: Distance cutoff for interface residue detection

    Returns:
        Tuple of (abmpnn_sequences, success_flag)
            - abmpnn_sequences: List of AbMPNN redesigned sequences
            - success_flag: Boolean indicating if redesign was successful
    """
    try:
        abmpnn_sequences = get_abmpnn_sequences(
            trajectory_pdb_af=trajectory_pdb_af,
            run_settings=run_settings,
            cdr_positions=run_settings["cdr_positions"],
            atom_distance_cutoff=atom_distance_cutoff,
            seed=seed,
        )

        success = len(abmpnn_sequences) > 0
        if not success:
            print("AbMPNN redesign failed: no sequences generated")

        return abmpnn_sequences, success

    except Exception:
        # Fail-fast: an AbMPNN failure here means either OOM, a config bug, or
        # genuine model breakage — none of which are safe to silently skip.
        # Previous behavior (return [], False) made the trajectory loop continue
        # to the next seed, hiding real bugs and wasting GPU time. Re-raise so
        # SLURM marks the whole job FAILED and the user investigates.
        import traceback
        print(
            f"\n{'=' * 78}\n"
            f"[ABMPNN PIPELINE FATAL] redesign failed; aborting the run "
            f"so the underlying issue can be diagnosed.\n"
            f"{'=' * 78}",
            flush=True,
        )
        traceback.print_exc()
        raise
