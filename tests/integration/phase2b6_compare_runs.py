#!/usr/bin/env python3
"""Compare the two frozen Phase 2B.6 production-pipeline executions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def normalized_config(path: Path, root: Path) -> dict:
  text = path.read_text()
  replacements = [str(root)]
  try:
    replacements.append(str(root.relative_to(Path.cwd())))
  except ValueError:
    pass
  for value in replacements:
    text = text.replace(value, "<RUN_ROOT>")
  values = {}
  for line in text.splitlines():
    if ": " in line and not line.lstrip().startswith("-"):
      key, value = line.strip().split(": ", 1)
      values[key] = value
  return values


def parse_log(path: Path) -> dict:
  text = path.read_text(errors="replace")
  return {
      "af2_iteration_records": re.findall(
          r"^\d+ models \[.*$", text, flags=re.MULTILINE
      ),
      "sequence_records": re.findall(
          r"^Sequence: ([A-Z]+)$", text, flags=re.MULTILINE
      ),
      "worst_sample_records": re.findall(
          r"^Selected worst sample for (.+?): (seed-\d+_sample-\d+) "
          r"\(ranking_score=([0-9.]+)\)$",
          text,
          flags=re.MULTILINE,
      ),
      "abmpnn_records": re.findall(
          r"^  Sequence \d+: score=([0-9.]+), seqid=([0-9.]+)$",
          text,
          flags=re.MULTILINE,
      ),
      "runtime_line": next(
          (line.strip() for line in text.splitlines() if line.startswith("Elapsed:")),
          None,
      ),
      "wall_time_line": next(
          (
              line.strip()
              for line in text.splitlines()
              if re.match(r"^\s*[0-9.]+ real\s", line)
          ),
          None,
      ),
  }


def load_csv(path: Path, run_root: Path) -> list[dict]:
  normalized = []
  replacements = [str(run_root)]
  try:
    replacements.append(str(run_root.relative_to(Path.cwd())))
  except ValueError:
    pass
  with path.open(newline="") as handle:
    for row in csv.DictReader(handle):
      normalized_row = {}
      for key, value in row.items():
        if key == "design_time":
          continue
        if isinstance(value, str):
          for replacement in replacements:
            value = value.replace(replacement, "<RUN_ROOT>")
        normalized_row[key] = value
      normalized.append(normalized_row)
  return normalized


def numeric_comparison(rows1: list[dict], rows2: list[dict]) -> dict:
  if len(rows1) != len(rows2):
    return {"row_count_equal": False}
  common = sorted(set(rows1[0]) & set(rows2[0])) if rows1 else []
  exact_fields = []
  differing_fields = []
  max_abs_by_field = {}
  for field in common:
    values1 = [row[field] for row in rows1]
    values2 = [row[field] for row in rows2]
    if values1 == values2:
      exact_fields.append(field)
      continue
    differing_fields.append(field)
    numeric = True
    differences = []
    for left, right in zip(values1, values2):
      try:
        differences.append(abs(float(left) - float(right)))
      except (TypeError, ValueError):
        numeric = False
        break
    if numeric:
      max_abs_by_field[field] = max(differences) if differences else 0.0
  return {
      "row_count_equal": True,
      "exact_field_count": len(exact_fields),
      "exact_fields": exact_fields,
      "differing_fields": differing_fields,
      "max_abs_numeric_difference_by_field": max_abs_by_field,
  }


def structure_inventory(path: Path, run_root: Path) -> dict:
  inventory = {}
  if not path.is_dir():
    return inventory
  for file_path in sorted(path.rglob("*")):
    if file_path.is_file():
      relative = str(file_path.relative_to(run_root))
      inventory[relative] = {
          "sha256": sha256(file_path),
          "size_bytes": file_path.stat().st_size,
      }
  return inventory


def compare_inventory(left: dict, right: dict) -> dict:
  common = sorted(set(left) & set(right))
  exact = [key for key in common if left[key]["sha256"] == right[key]["sha256"]]
  differing = [key for key in common if left[key]["sha256"] != right[key]["sha256"]]
  return {
      "left_count": len(left),
      "right_count": len(right),
      "common_count": len(common),
      "exact_count": len(exact),
      "all_common_exact": not differing,
      "only_left": sorted(set(left) - set(right)),
      "only_right": sorted(set(right) - set(left)),
      "differing": differing,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run1-root", type=Path, required=True)
  parser.add_argument("--run2-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  run1_root = args.run1_root.resolve()
  run2_root = args.run2_root.resolve()
  inner1 = run1_root / "results/germinal_run"
  inner2 = run2_root / "results/germinal_run"

  config1 = normalized_config(inner1 / "final_config.yaml", run1_root)
  config2 = normalized_config(inner2 / "final_config.yaml", run2_root)
  log1 = parse_log(run1_root.parent / "run1.log")
  log2 = parse_log(run2_root.parent / "run2.log")

  csv1_path = inner1 / "redesign_candidates/designs.csv"
  csv2_path = inner2 / "redesign_candidates/designs.csv"
  rows1 = load_csv(csv1_path, run1_root)
  rows2 = load_csv(csv2_path, run2_root)

  af3_inputs1 = structure_inventory(
      inner1 / "trajectories/structures/af3_inputs", run1_root
  )
  af3_inputs2 = structure_inventory(
      inner2 / "trajectories/structures/af3_inputs", run2_root
  )
  structures1 = structure_inventory(run1_root, run1_root)
  structures2 = structure_inventory(run2_root, run2_root)

  # Compare only corresponding AF3 inputs, not runtime log files.
  af3_json1 = {
      key: value for key, value in af3_inputs1.items() if key.endswith(".json")
  }
  af3_json2 = {
      key: value for key, value in af3_inputs2.items() if key.endswith(".json")
  }
  pdb1 = {key: value for key, value in structures1.items() if key.endswith(".pdb")}
  pdb2 = {key: value for key, value in structures2.items() if key.endswith(".pdb")}

  report = {
      "status": "PASS",
      "run1_root": str(run1_root),
      "run2_root": str(run2_root),
      "configuration": {
          "normalized_equal": config1 == config2,
          "differences": {
              key: [config1.get(key), config2.get(key)]
              for key in sorted(set(config1) | set(config2))
              if config1.get(key) != config2.get(key)
          },
      },
      "af2": {
          "iteration_record_count": [
              len(log1["af2_iteration_records"]),
              len(log2["af2_iteration_records"]),
          ],
          "iteration_records_equal": (
              log1["af2_iteration_records"] == log2["af2_iteration_records"]
          ),
          "sequence_record_count": [
              len(log1["sequence_records"]),
              len(log2["sequence_records"]),
          ],
          "sequence_records_equal": (
              log1["sequence_records"] == log2["sequence_records"]
          ),
          "final_sequence": log1["sequence_records"][-1],
          "pdb_hash_equal": (
              pdb1.get(
                  "results/germinal_run/trajectories/structures/"
                  "il3_nb_s1791095845.pdb"
              )
              == pdb2.get(
                  "results/germinal_run/trajectories/structures/"
                  "il3_nb_s1791095845.pdb"
              )
          ),
          "logits_distograms_parameter_state_note": (
              "The literal production entry point does not persist these "
              "internal AF2 tensors. Its 105 iteration records and final "
              "sequence/PDB are compared; tensor-level bitwise evidence is "
              "provided by the accepted Phase 2B.4 trace."
          ),
      },
      "abmpnn": {
          "records_equal": log1["abmpnn_records"] == log2["abmpnn_records"],
          "records": log1["abmpnn_records"],
      },
      "af3": {
          "worst_sample_records_equal": (
              log1["worst_sample_records"] == log2["worst_sample_records"]
          ),
          "worst_sample_records": log1["worst_sample_records"],
          "input_json_inventory": compare_inventory(af3_json1, af3_json2),
          "pdb_inventory": compare_inventory(pdb1, pdb2),
      },
      "filters": numeric_comparison(rows1, rows2),
      "run_summary": {
          "run1": {
              "runtime": log1["runtime_line"],
              "wall_time": log1["wall_time_line"],
          },
          "run2": {
              "runtime": log2["runtime_line"],
              "wall_time": log2["wall_time_line"],
          },
      },
  }

  checks = [
      report["configuration"]["normalized_equal"],
      report["af2"]["iteration_records_equal"],
      report["af2"]["sequence_records_equal"],
      report["abmpnn"]["records_equal"],
      report["af3"]["worst_sample_records_equal"],
      report["af3"]["input_json_inventory"]["all_common_exact"],
      report["af3"]["pdb_inventory"]["all_common_exact"],
      report["filters"].get("row_count_equal", False),
      not report["filters"].get("differing_fields", ["not_compared"]),
  ]
  report["status"] = "PASS" if all(checks) else "PARTIAL"
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
