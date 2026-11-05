"""Regression tests for the production AbMPNN process and numerical boundaries."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from germinal.filters import redesign


class AbMPNNBoundaryTests(unittest.TestCase):
    def test_finite_output_is_accepted(self):
        self.assertTrue(
            redesign._abmpnn_output_is_finite(
                {
                    "seq": np.array(["ACDE"]),
                    "score": np.array([1.0]),
                    "logits": np.zeros((1, 4, 21), dtype=np.float32),
                }
            )
        )

    def test_nan_score_is_rejected(self):
        self.assertFalse(
            redesign._abmpnn_output_is_finite(
                {
                    "seq": np.array(["AAAA"]),
                    "score": np.array([np.nan]),
                    "logits": np.zeros((1, 4, 21), dtype=np.float32),
                }
            )
        )

    def test_nan_logits_are_rejected(self):
        self.assertFalse(
            redesign._abmpnn_output_is_finite(
                {
                    "seq": np.array(["AAAA"]),
                    "score": np.array([1.0]),
                    "logits": np.full((1, 4, 21), np.nan),
                }
            )
        )

    def test_dedicated_worker_entrypoint_exists(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_abmpnn_worker.py"
        )
        self.assertTrue(path.is_file())

    def test_nonfinite_worker_result_is_retried(self):
        calls = []
        valid_sequences = [
            ("A" * 130) + amino_acid
            for amino_acid in ("C", "D", "E", "F", "G")
        ]

        def fake_run(command, **kwargs):
            calls.append(command)
            output_path = Path(
                command[command.index("--output-pickle") + 1]
            )
            if len(calls) == 1:
                payload = {
                    "seq": np.array(["A" * 131]),
                    "score": np.array([np.nan]),
                    "logits": np.full((1, 131, 21), np.nan),
                }
            else:
                payload = {
                    "seq": np.array(valid_sequences),
                    "score": np.arange(5, dtype=float) / 10,
                    "seqid": np.arange(5, dtype=float) / 100,
                    "logits": np.zeros((5, 131, 21)),
                }
            with output_path.open("wb") as handle:
                pickle.dump(payload, handle)
            return subprocess.CompletedProcess(command, 0)

        run_settings = {
            "cdr_positions": [],
            "max_mpnn_sequences": 5,
        }
        with (
            mock.patch.object(
                redesign,
                "get_sequence_from_pdb",
                return_value={"A": "K" * 112, "B": "A" * 131},
            ),
            mock.patch.object(redesign, "hotspot_residues", return_value={}),
            mock.patch.object(redesign.subprocess, "run", side_effect=fake_run),
            mock.patch.dict(
                os.environ,
                {"GERMINAL_ABMPNN_MAX_TRIALS": "2"},
            ),
        ):
            result = redesign.get_abmpnn_sequences(
                "boundary.pdb",
                run_settings,
                cdr_positions=[],
                atom_distance_cutoff=0.0,
                seed=1791095845,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result), 5)
        self.assertTrue(
            all(np.isfinite(float(item["score"])) for item in result)
        )

    def test_configured_runtime_identity_is_validated(self):
        invalid_sequences = [
            ("A" * 130) + amino_acid
            for amino_acid in ("C", "D", "E", "F", "G")
        ]

        def fake_run(command, **kwargs):
            output_path = Path(
                command[command.index("--output-pickle") + 1]
            )
            identity_path = Path(
                command[command.index("--runtime-identity") + 1]
            )
            payload = {
                "seq": np.array(invalid_sequences),
                "score": np.arange(5, dtype=float) / 10,
                "seqid": np.arange(5, dtype=float) / 100,
                "logits": np.zeros((5, 131, 21)),
            }
            with output_path.open("wb") as handle:
                pickle.dump(payload, handle)
            identity_path.write_text(
                json.dumps(
                    {
                        "sys_prefix": str(Path(sys.prefix).resolve()),
                        "jax_mps_version": "test-version",
                        "jax_default_backend": "mps",
                    }
                )
            )
            return subprocess.CompletedProcess(command, 0)

        run_settings = {
            "cdr_positions": [],
            "max_mpnn_sequences": 5,
            "abmpnn_python_path": sys.executable,
            "abmpnn_expected_jax_mps_version": "test-version",
            "abmpnn_preserve_worker_logs": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_settings["abmpnn_worker_log_dir"] = str(Path(tmp) / "workers")
            with (
                mock.patch.object(
                    redesign,
                    "get_sequence_from_pdb",
                    return_value={"A": "K" * 112, "B": "A" * 131},
                ),
                mock.patch.object(
                    redesign, "hotspot_residues", return_value={}
                ),
                mock.patch.object(
                    redesign.subprocess, "run", side_effect=fake_run
                ),
            ):
                result = redesign.get_abmpnn_sequences(
                    "boundary.pdb",
                    run_settings,
                    cdr_positions=[],
                    atom_distance_cutoff=0.0,
                    seed=1791095845,
                )

            self.assertEqual(len(result), 5)
            self.assertTrue(
                (Path(tmp) / "workers" / "run_settings.json").is_file()
            )
            self.assertTrue(
                (Path(tmp) / "workers" / "trial_01.log").is_file()
            )
            self.assertTrue(
                (
                    Path(tmp)
                    / "workers"
                    / "trial_01.runtime.json"
                ).is_file()
            )

    def test_wrong_runtime_version_fails_closed(self):
        valid_sequences = [
            ("A" * 130) + amino_acid
            for amino_acid in ("C", "D", "E", "F", "G")
        ]

        def fake_run(command, **kwargs):
            output_path = Path(
                command[command.index("--output-pickle") + 1]
            )
            identity_path = Path(
                command[command.index("--runtime-identity") + 1]
            )
            payload = {
                "seq": np.array(valid_sequences),
                "score": np.arange(5, dtype=float) / 10,
                "seqid": np.arange(5, dtype=float) / 100,
                "logits": np.zeros((5, 131, 21)),
            }
            with output_path.open("wb") as handle:
                pickle.dump(payload, handle)
            identity_path.write_text(
                json.dumps(
                    {
                        "sys_prefix": str(Path(sys.prefix).resolve()),
                        "jax_mps_version": "wrong-version",
                        "jax_default_backend": "mps",
                    }
                )
            )
            return subprocess.CompletedProcess(command, 0)

        run_settings = {
            "cdr_positions": [],
            "max_mpnn_sequences": 5,
            "abmpnn_python_path": sys.executable,
            "abmpnn_expected_jax_mps_version": "expected-version",
        }
        with (
            mock.patch.object(
                redesign,
                "get_sequence_from_pdb",
                return_value={"A": "K" * 112, "B": "A" * 131},
            ),
            mock.patch.object(redesign, "hotspot_residues", return_value={}),
            mock.patch.object(
                redesign.subprocess, "run", side_effect=fake_run
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "imported the wrong jax-mps version"
            ):
                redesign.get_abmpnn_sequences(
                    "boundary.pdb",
                    run_settings,
                    cdr_positions=[],
                    atom_distance_cutoff=0.0,
                    seed=1791095845,
                )


if __name__ == "__main__":
    unittest.main()
