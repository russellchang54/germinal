#!/usr/bin/env python3
"""Capture an immutable source and working-tree evidence snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SNAPSHOT_ROOTS = (
    "run_germinal.py",
    "germinal",
    "configs",
    "scripts",
    "colabdesign/colabdesign/mpnn",
)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git(*args: str) -> str:
  return subprocess.run(
      ["git", *args],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
  ).stdout


def iter_snapshot_files():
  for relative_root in SNAPSHOT_ROOTS:
    root = REPO_ROOT / relative_root
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
      if not path.is_file():
        continue
      relative = path.relative_to(REPO_ROOT)
      if "__pycache__" in relative.parts or path.suffix == ".pyc":
        continue
      yield path, relative


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--label", required=True)
  args = parser.parse_args()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  status = git("status", "--porcelain=v1")
  untracked = git("ls-files", "--others", "--exclude-standard")
  tracked_diff = git("diff", "--binary", "HEAD")
  staged_diff = git("diff", "--binary", "--cached", "HEAD")

  (output_dir / "git_status_porcelain.txt").write_text(status)
  (output_dir / "untracked_files.txt").write_text(untracked)
  (output_dir / "tracked_working_tree.diff").write_text(tracked_diff)
  (output_dir / "staged.diff").write_text(staged_diff)

  files = []
  snapshot_path = output_dir / "source_snapshot.tar.gz"
  with tarfile.open(snapshot_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
    for path, relative in iter_snapshot_files():
      info = archive.gettarinfo(str(path), arcname=str(relative))
      info.uid = 0
      info.gid = 0
      info.uname = ""
      info.gname = ""
      payload = path.read_bytes()
      import io

      info.size = len(payload)
      archive.addfile(info, io.BytesIO(payload))
      files.append(
          {
              "path": str(relative),
              "size_bytes": path.stat().st_size,
              "sha256": sha256(path),
          }
      )

  manifest = {
      "label": args.label,
      "git_commit": git("rev-parse", "HEAD").strip(),
      "status_line_count": len(status.splitlines()),
      "untracked_file_count": len(untracked.splitlines()),
      "tracked_working_tree_diff": {
          "path": str(output_dir / "tracked_working_tree.diff"),
          "size_bytes": (output_dir / "tracked_working_tree.diff").stat().st_size,
          "sha256": sha256(output_dir / "tracked_working_tree.diff"),
      },
      "staged_diff": {
          "path": str(output_dir / "staged.diff"),
          "size_bytes": (output_dir / "staged.diff").stat().st_size,
          "sha256": sha256(output_dir / "staged.diff"),
      },
      "source_snapshot": {
          "path": str(snapshot_path),
          "size_bytes": snapshot_path.stat().st_size,
          "sha256": sha256(snapshot_path),
          "file_count": len(files),
          "files": files,
      },
  }
  output_path = output_dir / "baseline.json"
  output_path.write_text(json.dumps(manifest, indent=2) + "\n")

  for immutable_path in (
      snapshot_path,
      output_path,
      output_dir / "git_status_porcelain.txt",
      output_dir / "untracked_files.txt",
      output_dir / "tracked_working_tree.diff",
      output_dir / "staged.diff",
  ):
    os.chmod(immutable_path, 0o444)

  print(json.dumps(manifest, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
