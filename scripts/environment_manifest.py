#!/usr/bin/env python3
"""Write a deterministic manifest of installed Python distributions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages[name.lower()] = {
            "name": name,
            "version": distribution.version,
            "location": str(Path(distribution.locate_file("")).resolve()),
            "direct_url": _optional_json(
                distribution.read_text("direct_url.json")
            ),
        }
    manifest = {
        "python_executable": str(Path(sys.executable)),
        "python_version": sys.version,
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_count": len(packages),
        "packages": dict(sorted(packages.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "sys_prefix": manifest["sys_prefix"],
                "package_count": manifest["package_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


def _optional_json(value: str | None):
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
