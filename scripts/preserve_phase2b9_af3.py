#!/usr/bin/env python3
"""Clone completed AF3 job directories before Germinal removes them."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path


def main() -> int:
    run_root = Path(
        "docs/porting/af3_mps_probe/results/phase2b9/run2"
    ).resolve()
    structures = run_root / "results/germinal_run/trajectories/structures"
    destination = structures / "af3_outputs_preserved"
    destination.mkdir(parents=True, exist_ok=True)
    log = destination / "preservation.log"
    while subprocess.run(
        ["pgrep", "-f", "run_germinal.py.*phase2b9/run2"],
        capture_output=True,
    ).returncode == 0:
        if structures.is_dir():
            for source in structures.glob("il3_nb_s1791095845*"):
                if not source.is_dir() or source == destination:
                    continue
                model = source / f"{source.name}_model.cif"
                if not model.is_file():
                    continue
                target = destination / source.name
                if target.exists():
                    continue
                temporary = destination / f".{source.name}.copying"
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(source, temporary, copy_function=shutil.copy2)
                temporary.rename(target)
                with log.open("a") as handle:
                    handle.write(f"preserved {source.name}\n")
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
