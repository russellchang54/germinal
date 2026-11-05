#!/usr/bin/env python3
"""Launch AF3 from an explicit checkout, bypassing stale editable metadata."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--af3-source", required=True)
    args, af3_args = parser.parse_known_args()
    source = Path(args.af3_source).expanduser().resolve()
    source_root = source / "src"
    run_script = source / "run_alphafold.py"
    if not run_script.is_file():
        raise SystemExit(f"AF3 run_alphafold.py not found: {run_script}")

    # uv installed this checkout as an editable package while it lived in
    # /tmp. Scikit-build's redirecting finder precedes normal path lookup, so
    # remove it before importing alphafold3 from the configured persistent path.
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not type(finder).__module__.startswith("_editable_skbc_alphafold3")
    ]
    sys.path.insert(0, str(source_root))
    import alphafold3  # pylint: disable=import-outside-toplevel

    imported = Path(alphafold3.__file__).resolve()
    try:
        imported.relative_to(source_root.resolve())
    except ValueError as exc:
        raise SystemExit(
            "alphafold3 imports from an unexpected checkout: "
            f"{imported}; expected within {source_root.resolve()}"
        ) from exc
    print(f"[af3-native] Source import verified: {imported}", flush=True)

    sys.argv = [str(run_script), *af3_args]
    runpy.run_path(str(run_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
