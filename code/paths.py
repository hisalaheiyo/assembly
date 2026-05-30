"""Centralized path configuration. All paths are overridable via environment
variables so the code can run from any layout without touching source.

Defaults assume a flat layout under the repo root:

    ./
      data/{meccano,havid,impact}/   <-- dataset roots (download separately)
      outputs/
        checkpoints/                 <-- saved .pth files
        results/                     <-- per-config eval JSON
        tables/                      <-- generated .tex table files
        figures/                     <-- generated .pdf figure files

Override any path by exporting the corresponding environment variable before
running, e.g.:

    export PROJECT_DIR=/my/outputs
    export MECCANO_DATA=/data/MECCANO_release
    export HAVID_DATA=/data/HAViD_view2
    export IMPACT_DATA=/data/IMPACT_release
"""
import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


# Project root for all generated outputs.
PROJECT_DIR     = _env_path("PROJECT_DIR",     "./outputs")

# Dataset roots (download separately; see data/README.md).
MECCANO_DATA    = _env_path("MECCANO_DATA",    "./data/meccano")
HAVID_DATA      = _env_path("HAVID_DATA",      "./data/havid")
IMPACT_DATA     = _env_path("IMPACT_DATA",     "./data/impact")

# Per-purpose output directories (defaults nest under PROJECT_DIR).
CHECKPOINTS_DIR = _env_path("CHECKPOINTS_DIR", str(PROJECT_DIR / "checkpoints"))
RESULTS_DIR     = _env_path("RESULTS_DIR",     str(PROJECT_DIR / "results"))
TABLES_DIR      = _env_path("TABLES_DIR",      str(PROJECT_DIR / "tables"))
FIGURES_DIR     = _env_path("FIGURES_DIR",     str(PROJECT_DIR / "figures"))
FONTS_DIR       = _env_path("FONTS_DIR",       str(PROJECT_DIR / "fonts"))

# Optional CSV-prefix remap for legacy MECCANO indices that contain absolute
# paths from another machine. Set MECCANO_CSV_OLD_PREFIX to the literal prefix
# to strip, MECCANO_CSV_NEW_PREFIX to what it should become (defaults to
# MECCANO_DATA). Leave OLD empty to disable remap.
MECCANO_CSV_OLD_PREFIX = os.environ.get("MECCANO_CSV_OLD_PREFIX", "")
MECCANO_CSV_NEW_PREFIX = os.environ.get("MECCANO_CSV_NEW_PREFIX", str(MECCANO_DATA))
