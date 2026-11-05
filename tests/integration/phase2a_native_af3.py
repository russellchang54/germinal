#!/usr/bin/env python3
"""Run one Germinal-generated VHH/IL3 AF3 job through native Apple MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from Bio import PDB

from germinal.filters import af3
from germinal.utils.utils import get_sequence_from_pdb


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / ".envs/af3-mps-probe/share/phase2a-germinal-af3",
    )
    parser.add_argument("--design-name", default="phase2a_vhh_il3_native")
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_pdb = output_dir / f"{args.design_name.lower()}_af3.pdb"
    if expected_pdb.exists():
        parser.error(
            f"Integration output already exists: {expected_pdb}. "
            "Choose a new --design-name rather than overwriting a prior run."
        )

    target_sequences = get_sequence_from_pdb(str(REPO_ROOT / "pdbs/il3.pdb"))
    binder_sequences = get_sequence_from_pdb(str(REPO_ROOT / "pdbs/nb.pdb"))
    target_sequence = target_sequences["A"]
    binder_sequence = binder_sequences["A"]

    run_settings = {
        "af3_execution_backend": "native_mps",
        "af3_expected_revision": af3.AF3_EXPECTED_REVISION,
        "af3_repo_path": ".envs/af3-mps-probe/share/af3-src",
        "af3_model_dir": ".envs/af3-mps-probe/share/af3-models",
        "af3_db_dir": None,
        "af3_database_manifest": None,
        "af3_python_path": ".envs/af3-mps-probe/bin/python",
        "af3_tools_bin_dir": ".envs/af3-mps-tools/bin",
        "af3_runner_path": "scripts/run_af3_mps.py",
        "af3_tokamax_patch_path": (
            "docs/porting/af3_mps_probe/patches/"
            "tokamax-0.0.12-mps-f32.patch"
        ),
        "af3_run_data_pipeline": False,
        "af3_num_diffusion_samples": args.num_diffusion_samples,
        "af3_num_recycles": args.num_recycles,
        "af3_timeout_seconds": args.timeout_seconds,
        "msa_db_dir": None,
        "use_metagenomic_db": False,
    }

    started = time.perf_counter()
    pdb_path, scores, ipsae = af3.run_af3(
        binder_seq=binder_sequence,
        target_seq=target_sequence,
        target_chains="A",
        output_dir=str(output_dir),
        design_name=args.design_name,
        seed=[42],
        run_settings=run_settings,
        binder_chain="B",
        msa_mode="none",
        select_mode="best",
    )
    elapsed = time.perf_counter() - started

    structure = PDB.PDBParser(QUIET=True).get_structure("prediction", pdb_path)
    chain_ids = [chain.id for chain in structure[0]]
    residue_counts = {chain.id: len(chain) for chain in structure[0]}
    coordinates = np.asarray(
        [atom.coord for atom in structure.get_atoms()], dtype=np.float64
    )
    expected_shape = (
        len(target_sequence) + len(binder_sequence),
        len(target_sequence) + len(binder_sequence),
    )
    log_path = (
        output_dir
        / "af3_inputs"
        / f"{args.design_name}_af3_subprocess.log"
    )
    log_text = log_path.read_text(errors="replace")
    checks = {
        "pdb_exists": Path(pdb_path).is_file(),
        "chains": chain_ids == ["A", "B"],
        "target_residues": residue_counts.get("A") == len(target_sequence),
        "binder_residues": residue_counts.get("B") == len(binder_sequence),
        "coordinates_finite": bool(np.isfinite(coordinates).all()),
        "pae_shape": scores["pae_matrix"].shape == expected_shape,
        "pae_finite": bool(np.isfinite(scores["pae_matrix"]).all()),
        "iptm_present": bool(scores["iptm"] and np.isfinite(scores["iptm"]).all()),
        "plddt_present": bool(
            scores["plddt"] is not None and np.isfinite(scores["plddt"])
        ),
        "mps_selected": "default_backend=mps" in log_text
        and "device=MPS:0" in log_text,
        "no_cpu_platform_override": "JAX_PLATFORMS=cpu" not in log_text,
        "particle_metal_requested": "Found local MPS devices" in log_text,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "design_name": args.design_name,
        "sequence_handoff": {
            "target_chain": "A",
            "target_residues": len(target_sequence),
            "target_sha256": _sha256(target_sequence),
            "binder_chain": "B",
            "binder_residues": len(binder_sequence),
            "binder_sha256": _sha256(binder_sequence),
        },
        "integration_controls": {
            "seed": 42,
            "num_diffusion_samples": args.num_diffusion_samples,
            "num_recycles": args.num_recycles,
            "msa_mode": "none",
            "run_data_pipeline": False,
            "scope": (
                "execution-boundary integration only; not a production-database "
                "or full Germinal filter-run result"
            ),
        },
        "runtime_seconds": round(elapsed, 6),
        "max_rss_bytes_this_process": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "artifacts": {
            "pdb": str(Path(pdb_path).resolve()),
            "subprocess_log": str(log_path.resolve()),
            "af3_input": str(
                (output_dir / "af3_inputs" / f"{args.design_name}.json").resolve()
            ),
        },
        "output": {
            "chains": chain_ids,
            "residue_counts": residue_counts,
            "atoms": int(coordinates.shape[0]),
            "plddt": float(scores["plddt"]),
            "plddt_binder": float(scores["plddt_binder"]),
            "ptm": scores["ptm"],
            "iptm": scores["iptm"],
            "pae_mean": float(scores["pae"]),
            "binder_pae_mean": float(scores["binder_pae"]),
            "pae_matrix_shape": list(scores["pae_matrix"].shape),
            "aggregate_score": scores["aggregate_score"],
            "ipsae": ipsae,
        },
        "checks": checks,
    }
    report_path = output_dir / "phase2a_integration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
