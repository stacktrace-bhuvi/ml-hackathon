"""API settings (environment variables). Nothing about any trip's result is hardcoded.

PATHSNAP_TRIPS_ROOT  folder that contains the split folder           (default: data/raw/Student folder)
PATHSNAP_SPLIT       split folder whose trips are offered for evaluation (default: student_test)
                     On evaluation day put the 10 evaluator trips in e.g. data/raw/eval/evaluator_test/
                     and set PATHSNAP_TRIPS_ROOT=data/raw/eval  PATHSNAP_SPLIT=evaluator_test
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIPS_ROOT = Path(os.getenv("PATHSNAP_TRIPS_ROOT", ROOT / "data" / "raw" / "Student folder"))
SPLIT = os.getenv("PATHSNAP_SPLIT", "student_test")
