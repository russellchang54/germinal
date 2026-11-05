"""Integration-boundary tests for native AF3 execution without running models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir

from germinal.filters import af3


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_RUNNER = REPO_ROOT / "scripts" / "run_af3_mps.py"


def _load_native_runner():
    spec = importlib.util.spec_from_file_location("run_af3_mps", NATIVE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AF3BoundaryUnitTests(unittest.TestCase):
    def test_create_input_dict_preserves_af3_semantics(self):
        job = af3.create_input_dict(
            "ACDE",
            "FGHI",
            "B",
            "A",
            "integration_job",
            [42],
        )
        self.assertEqual(job["dialect"], "alphafold3")
        self.assertEqual(job["version"], 2)
        self.assertEqual(job["modelSeeds"], [42])
        self.assertEqual(
            [entry["protein"]["id"] for entry in job["sequences"]],
            [["B"], ["A"]],
        )
        self.assertEqual(
            [entry["protein"]["sequence"] for entry in job["sequences"]],
            ["ACDE", "FGHI"],
        )
        self.assertEqual(
            [
                {
                    key: entry["protein"][key]
                    for key in ("unpairedMsa", "pairedMsa", "templates")
                    if key in entry["protein"]
                }
                for entry in job["sequences"]
            ],
            [{}, {}],
        )

    def test_target_msa_uses_complete_custom_chain_specification(self):
        job = af3.create_input_dict(
            "ACDE",
            "FGHI",
            "B",
            "A",
            "integration_job",
            [42],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "msas").mkdir()
            (root / "msas" / "target_A.a3m").write_text(">query\nFGHI\n")
            with mock.patch.object(
                af3,
                "call_generate_colabfold_msa_with_timeout",
                return_value="",
            ):
                updated = af3.generate_msas(
                    job,
                    "",
                    str(root),
                    binder_chain="B",
                    msa_mode="target",
                )

        target = updated["sequences"][1]["protein"]
        self.assertEqual(target["unpairedMsaPath"], "msas/target_A.a3m")
        self.assertEqual(target["pairedMsa"], "")
        self.assertEqual(target["templates"], [])
        self.assertNotIn("unpairedMsa", target)
        self.assertNotIn("pairedMsa", updated["sequences"][0]["protein"])
        self.assertNotIn("templates", updated["sequences"][0]["protein"])

    def test_auto_selects_native_mps_on_macos(self):
        self.assertEqual(
            af3._select_execution_backend(
                {"af3_execution_backend": "auto"}, platform_name="Darwin"
            ),
            "native_mps",
        )

    def test_auto_preserves_singularity_on_linux(self):
        self.assertEqual(
            af3._select_execution_backend(
                {"af3_execution_backend": "auto"}, platform_name="Linux"
            ),
            "singularity",
        )

    def test_native_mps_is_rejected_off_macos(self):
        with self.assertRaises(af3.AF3ConfigurationError):
            af3._select_execution_backend(
                {"af3_execution_backend": "native_mps"},
                platform_name="Linux",
            )

    def test_native_command_contains_explicit_boundary_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_dir = root / "databases"
            db_dir.mkdir()
            command, timeout = af3._native_af3_command(
                str(root / "job.json"),
                str(root / "output"),
                {
                    "af3_python_path": sys.executable,
                    "af3_repo_path": str(root / "af3"),
                    "af3_model_dir": str(root / "models"),
                    "af3_db_dir": str(db_dir),
                    "af3_tools_bin_dir": str(root / "tools"),
                    "af3_tokamax_patch_path": str(root / "mps.patch"),
                    "af3_expected_revision": af3.AF3_EXPECTED_REVISION,
                    "af3_num_diffusion_samples": 2,
                    "af3_num_recycles": 7,
                    "af3_timeout_seconds": 123,
                },
            )
        self.assertEqual(timeout, 123)
        self.assertEqual(command[0], os.path.abspath(sys.executable))
        self.assertEqual(command[1], str(NATIVE_RUNNER))
        self.assertIn("--run-data-pipeline", command)
        self.assertIn("--num-diffusion-samples", command)
        self.assertIn("2", command)
        self.assertIn("--num-recycles", command)
        self.assertIn("7", command)
        self.assertIn(af3.AF3_EXPECTED_REVISION, command)

    def test_native_command_requires_database_for_data_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(
                af3.AF3ConfigurationError, "af3_db_dir"
            ):
                af3._native_af3_command(
                    str(root / "job.json"),
                    str(root / "output"),
                    {
                        "af3_python_path": sys.executable,
                        "af3_repo_path": str(root / "af3"),
                        "af3_model_dir": str(root / "models"),
                        "af3_tools_bin_dir": str(root / "tools"),
                        "af3_tokamax_patch_path": str(root / "mps.patch"),
                    },
                )

    def test_linux_singularity_command_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = af3._singularity_af3_command(
                str(root / "af3_inputs" / "job.json"),
                str(root / "af3_inputs"),
                str(root / "output"),
                {
                    "af3_repo_path": str(root / "af3"),
                    "af3_model_dir": str(root / "models"),
                    "af3_db_dir": str(root / "db"),
                    "af3_sif_path": str(root / "af3.sif"),
                },
            )
        self.assertEqual(command[:3], ["singularity", "exec", "--nv"])
        self.assertIn("/root/alphafold3/run_alphafold.py", command)
        self.assertIn("--db_dir=/root/public_databases", command)

    def test_input_handoff_writes_json_and_invokes_native_runner(self):
        job = af3.create_input_dict("ACDE", "FGHI", "B", "A", "job_1", [42])
        expected_result = (Path("result.pdb"), {"plddt": 1.0}, None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {
                "af3_execution_backend": "native_mps",
                "af3_python_path": sys.executable,
                "af3_repo_path": str(root / "af3"),
                "af3_model_dir": str(root / "models"),
                "af3_tools_bin_dir": str(root / "tools"),
                "af3_tokamax_patch_path": str(root / "mps.patch"),
                "af3_run_data_pipeline": False,
            }
            with mock.patch.object(
                af3, "_run_logged_subprocess"
            ) as run_mock, mock.patch.object(
                af3,
                "extract_structure_and_scores",
                return_value=expected_result,
            ):
                result = af3._run_af3(
                    job,
                    str(root),
                    binder_chain="B",
                    msa_mode="none",
                    run_settings=settings,
                )

            input_path = root / "af3_inputs" / "job_1.json"
            written = json.loads(input_path.read_text())
            command = run_mock.call_args.args[0]
            log_path = run_mock.call_args.kwargs["log_path"]

        self.assertEqual(result, expected_result)
        self.assertEqual(written, job)
        self.assertEqual(command[1], str(NATIVE_RUNNER))
        self.assertTrue(log_path.endswith("job_1_af3_subprocess.log"))

    def test_best_output_discovery_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                FileNotFoundError, "expected output artifacts"
            ):
                af3._af3_output_paths(tmp, "missing_job", select_mode="best")

    def test_best_output_discovery_finds_standard_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "job_1"
            folder.mkdir()
            expected = (
                folder / "job_1_model.cif",
                folder / "job_1_summary_confidences.json",
                folder / "job_1_confidences.json",
            )
            for path in expected:
                path.touch()
            actual = af3._af3_output_paths(tmp, "JOB_1", select_mode="best")
        self.assertEqual(actual, tuple(str(path) for path in expected))

    def test_subprocess_failure_propagates_exit_code_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = str(Path(tmp) / "failure.log")
            with self.assertRaises(af3.AF3ExecutionError) as raised:
                af3._run_logged_subprocess(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    log_path=log_path,
                )
            self.assertEqual(raised.exception.returncode, 7)
            self.assertEqual(raised.exception.log_path, log_path)
            self.assertIn(log_path, str(raised.exception))
            self.assertTrue(Path(log_path).is_file())

    def test_subprocess_timeout_kills_process_and_names_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = str(Path(tmp) / "timeout.log")
            with self.assertRaises(af3.AF3TimeoutError) as raised:
                af3._run_logged_subprocess(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    log_path=log_path,
                    timeout_seconds=0.1,
                )
            self.assertEqual(raised.exception.log_path, log_path)
            self.assertIn(log_path, str(raised.exception))


class NativeRunnerUnitTests(unittest.TestCase):
    def test_database_manifest_builds_explicit_af3_flags(self):
        native = _load_native_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {}
            for key in native.DATABASE_FLAGS:
                target = root / key
                target.mkdir()
                manifest[key] = target.name
            manifest_path = root / "databases.json"
            manifest_path.write_text(json.dumps(manifest))
            args = argparse.Namespace(
                run_data_pipeline=True,
                db_dir=None,
                database_manifest=str(manifest_path),
            )
            command = native._database_arguments(args)
        self.assertEqual(len(command), 18)
        for flag in native.DATABASE_FLAGS.values():
            self.assertIn(flag, command)

    def test_runner_contains_explicit_mps_flags(self):
        source = NATIVE_RUNNER.read_text()
        self.assertIn("--jax_backend=mps", source)
        self.assertIn("--flash_attention_implementation=xla", source)
        self.assertIn('env["JAX_PLATFORMS"] = "mps"', source)
        self.assertIn("run_af3_from_source.py", source)


class NativeConfigurationTests(unittest.TestCase):
    def test_macos_run_config_inherits_scientific_settings_unchanged(self):
        config_dir = str(REPO_ROOT / "configs")
        with initialize_config_dir(
            config_dir=config_dir, version_base=None
        ):
            base = compose(
                config_name="config",
                overrides=[
                    "run=vhh_il3",
                    "target=il3",
                    "filter/initial=vhh_il3",
                    "filter/final=vhh_il3",
                ],
            )
        with initialize_config_dir(
            config_dir=config_dir, version_base=None
        ):
            native = compose(
                config_name="config",
                overrides=[
                    "run=vhh_il3_macos",
                    "target=il3",
                    "filter/initial=vhh_il3",
                    "filter/final=vhh_il3",
                ],
            )
        scientific_keys = (
            "structure_model",
            "msa_mode",
            "af3_structure_select_mode",
            "ablm_model",
            "plddt_threshold",
            "i_ptm_threshold",
            "i_pae_threshold",
            "num_seqs",
            "max_mpnn_sequences",
        )
        for key in scientific_keys:
            self.assertEqual(native[key], base[key], key)
        self.assertEqual(native.af3_execution_backend, "native_mps")
        self.assertEqual(
            native.af3_expected_revision, af3.AF3_EXPECTED_REVISION
        )


if __name__ == "__main__":
    unittest.main()
