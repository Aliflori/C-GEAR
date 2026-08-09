#!/usr/bin/env python
"""CLI entry point for matched dynamic-rank parameter reporting."""

import importlib.util
from pathlib import Path


NLU_ROOT = Path(__file__).resolve().parents[2]
REPORTING_MODULE = NLU_ROOT / "src" / "transformers" / "rank_budget_reporting.py"
SPEC = importlib.util.spec_from_file_location("rank_budget_reporting", REPORTING_MODULE)
REPORTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTING)


if __name__ == "__main__":
    REPORTING.main()
