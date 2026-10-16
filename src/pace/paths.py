"""Paths shared by source-checkout workflows and benchmark adapters."""

import os
from pathlib import Path

PROJECT_ROOT = (
    Path(os.environ.get("PACE_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    .expanduser()
    .resolve()
)
