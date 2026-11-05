#!/usr/bin/env python3
"""Run fresh-process beta_strand traces and compare their state boundaries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_SCRIPT = REPO_ROOT / "tests/integration/phase2b1_beta_trace.py"


def _load_trace(path: Path) -> dict:
  return json.loads((path / "trace.json").read_text())


def _all_equal(values: list) -> bool:
  return bool(values) and all(value == values[0] for value in values[1:])


def _pass_map(trace: dict) -> dict[str, dict]:
  return {
      f"step{item['step']}_pass{item['pass']:02d}": item
      for item in trace["single_passes"]
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", choices=["mps", "cpu"], required=True)
  parser.add_argument(
      "--beta-variant",
      choices=["original", "candidate", "top_k"],
      default="original",
  )
  parser.add_argument("--processes", type=int, default=4)
  parser.add_argument("--steps", type=int, default=2)
  parser.add_argument("--target-residues", type=int, default=31)
  parser.add_argument("--gradient-decomposition", action="store_true")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--initial-params",
      type=Path,
      default=REPO_ROOT
      / "docs/porting/af3_mps_probe/results/phase2b/mps_initial_params_1step.npy",
  )
  args = parser.parse_args()

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  traces = []
  for process_index in range(1, args.processes + 1):
    process_dir = output_dir / f"process_{process_index}"
    process_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"process_{process_index}.log"
    command = [
        str(REPO_ROOT / ".envs/germinal-mps-design/bin/python"),
        str(TRACE_SCRIPT),
        "--backend",
        args.backend,
        "--beta-variant",
        args.beta_variant,
        "--steps",
        str(args.steps),
        "--target-residues",
        str(args.target_residues),
        "--initial-params",
        str(args.initial_params.resolve()),
        "--output-dir",
        str(process_dir),
    ]
    if args.gradient_decomposition:
      command.append("--gradient-decomposition")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".envs/germinal-mps-design/share/mpl")
    with log_path.open("w") as log_file:
      subprocess.run(
          command,
          cwd=REPO_ROOT,
          stdout=log_file,
          stderr=subprocess.STDOUT,
          env=env,
          check=True,
      )
    trace = _load_trace(process_dir)
    trace["process"] = process_index
    traces.append(trace)
    (output_dir / f"trace_process_{process_index}.json").write_text(
        json.dumps(trace, indent=2) + "\n"
    )
  (output_dir / "raw_four_process_traces.json").write_text(
      json.dumps(
          {
              "backend": args.backend,
              "beta_variant": args.beta_variant,
              "processes": traces,
          },
          indent=2,
      )
      + "\n"
  )

  state_by_boundary = {}
  boundaries = [
      item["boundary"] for item in traces[0]["state_snapshots"]
  ]
  for boundary in boundaries:
    hashes = []
    for trace in traces:
      entry = next(
          item
          for item in trace["state_snapshots"]
          if item["boundary"] == boundary
      )
      hashes.append(entry["sha256"])
    state_by_boundary[boundary] = {
        "hashes": hashes,
        "identical": _all_equal(hashes),
    }

  pass_maps = [_pass_map(trace) for trace in traces]
  pass_comparison = {}
  for key in pass_maps[0]:
    loss_components = [
        {
            name: item[key]["loss_components"][name]["value"]
            for name in item[key]["loss_components"]
        }
        for item in pass_maps
    ]
    pass_comparison[key] = {
        "beta_values": [item[key]["beta_strand"] for item in pass_maps],
        "beta_identical": _all_equal(
            [item[key]["beta_strand"] for item in pass_maps]
        ),
        "loss_component_values": loss_components,
        "loss_components_identical": _all_equal(loss_components),
        "distogram_hashes": [
            item[key]["distogram_sha256"] for item in pass_maps
        ],
        "distograms_identical": _all_equal(
            [item[key]["distogram_sha256"] for item in pass_maps]
        ),
        "key_hashes": [item[key]["key_sha256"] for item in pass_maps],
        "keys_identical": _all_equal(
            [item[key]["key_sha256"] for item in pass_maps]
        ),
        "gradient_devices": [
            item[key]["gradient_device"] for item in pass_maps
        ],
    }

  initial_hashes = [trace["initial_params_sha256"] for trace in traces]
  gradient_decomposition = None
  if all("gradient_decomposition" in trace for trace in traces):
    decomposition = {}
    for key in [
        "beta_only",
        "nonbeta_germinal_and_af2",
        "af2_total",
    ]:
      entries = [trace["gradient_decomposition"][key] for trace in traces]
      decomposition[key] = {
          "hashes": [entry["sha256"] for entry in entries],
          "l2": [entry["l2"] for entry in entries],
          "max_abs": [entry["max_abs"] for entry in entries],
          "nonzero_count": [entry["nonzero_count"] for entry in entries],
          "identical": _all_equal([entry["sha256"] for entry in entries]),
      }
    iglm_entries = [
        trace["gradient_decomposition"]["iglm"] for trace in traces
    ]
    decomposition["iglm_raw"] = {
        "hashes": [entry["sha256"] for entry in iglm_entries],
        "l2": [entry["l2"] for entry in iglm_entries],
        "max_abs": [entry["max_abs"] for entry in iglm_entries],
        "nonzero_count": [
            entry["nonzero_count"] for entry in iglm_entries
        ],
        "identical": _all_equal(
            [entry["sha256"] for entry in iglm_entries]
        ),
    }
    gradient_decomposition = decomposition

  pre_pcgrad = None
  if all("pcgrad" in trace.get("actual_gradient_boundaries", {}) for trace in traces):
    pre_pcgrad = {}
    for key in ["af2", "iglm", "combined"]:
      entries = [
          trace["actual_gradient_boundaries"]["pcgrad"][key]
          for trace in traces
      ]
      hashes = [entry["sha256"] for entry in entries]
      pre_pcgrad[key] = {
          "hashes": hashes,
          "l2": [entry["l2"] for entry in entries],
          "max_abs": [entry["max_abs"] for entry in entries],
          "nonzero_count": [entry["nonzero_count"] for entry in entries],
          "identical": _all_equal(hashes),
      }
  report = {
      "backend": args.backend,
      "beta_variant": args.beta_variant,
      "processes": args.processes,
      "steps": args.steps,
      "target_residues": args.target_residues,
      "gradient_decomposition": args.gradient_decomposition,
      "initial_params_identical": _all_equal(initial_hashes),
      "initial_param_hashes": initial_hashes,
      "state_boundaries": state_by_boundary,
      "passes": pass_comparison,
      "step_metrics": [
          [
              {
                  "iteration": step["iteration"],
                  "loss": step["loss"],
                  "beta_strand": step["beta_strand"],
                  "gradient_l2": step["gradient_l2"],
                  "optimizer_update": step["optimizer_update"],
                  "post_pcgrad_logged": step["post_pcgrad_logged"],
                  "before_state_sha256": step["before_state_sha256"],
                  "after_state_sha256": step["after_state_sha256"],
              }
              for step in trace["steps"]
          ]
          for trace in traces
      ],
      "gradient_decomposition": gradient_decomposition,
      "pre_pcgrad": pre_pcgrad,
      "second_step_beta_values": [
          item["step2_pass05"]["beta_strand"] for item in pass_maps
      ],
      "second_step_distograms_identical": _all_equal(
          [
              item["step2_pass05"]["distogram_sha256"]
              for item in pass_maps
          ]
      ),
      "second_step_keys_identical": _all_equal(
          [item["step2_pass05"]["key_sha256"] for item in pass_maps]
      ),
      "trace_files": [
          str(output_dir / f"trace_process_{index}.json")
          for index in range(1, args.processes + 1)
      ],
  }
  report_path = output_dir / "repeatability.json"
  report_path.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
