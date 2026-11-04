"""
Master Test Runner for GoalGuard Comprehensive Evaluation Suite.

Discovers and executes:
- Unit tests (A)
- Integration tests (B)
- Negative & input validation tests (C)
- Model output tests (D)
- Temporal logic tests (E)
- Data leakage tests (F)
- Regression & reproducibility tests (G)
- Original root tests status check

Exports execution report to evaluation/results/test_report.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_comprehensive_tests() -> dict[str, Any]:
    print("=" * 70)
    print("RUNNING GOALGUARD COMPREHENSIVE SOFTWARE TESTING SUITE")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        "evaluation.tests.test_unit",
        "evaluation.tests.test_integration",
        "evaluation.tests.test_negative_validation",
        "evaluation.tests.test_model_outputs",
        "evaluation.tests.test_temporal_logic",
        "evaluation.tests.test_data_leakage",
        "evaluation.tests.test_regression_reproducibility",
    ]

    for mod in test_modules:
        try:
            suite.addTests(loader.loadTestsFromName(mod))
        except Exception as exc:
            print(f"[Error loading {mod}]: {exc}")

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time

    # Check original root tests execution status
    original_tests_status: dict[str, Any] = {}
    try:
        import langgraph
        original_tests_status["test_goalguard_local_agent.py"] = "AVAILABLE"
    except ImportError:
        original_tests_status["test_goalguard_local_agent.py"] = "BLOCKED: Requires langgraph (not installed in base Python)"

    try:
        import torch
        original_tests_status["test_train_goalguard_gru.py"] = "AVAILABLE"
        original_tests_status["test_train_goalguard_local_gru.py"] = "AVAILABLE"
    except ImportError:
        original_tests_status["test_train_goalguard_gru.py"] = "BLOCKED: Requires torch (not installed in base Python)"
        original_tests_status["test_train_goalguard_local_gru.py"] = "BLOCKED: Requires torch (not installed in base Python)"

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(duration, 3),
        "total_tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "failures_detail": [str(f[0]) + ": " + str(f[1]) for f in result.failures],
        "errors_detail": [str(e[0]) + ": " + str(e[1]) for e in result.errors],
        "original_root_tests": original_tests_status,
        "verdict": "PASSED" if (len(result.failures) == 0 and len(result.errors) == 0) else "FAILED",
    }

    results_dir = REPO_ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Total Tests Run: {report['total_tests_run']}")
    print(f"Passed:          {report['passed']}")
    print(f"Failed:          {report['failed']}")
    print(f"Errors:          {report['errors']}")
    print(f"Skipped:         {report['skipped']}")
    print(f"Verdict:         {report['verdict']}")
    print("-" * 70)
    print("Original Root Tests Status:")
    for k, v in original_tests_status.items():
        print(f"  - {k}: {v}")
    print("=" * 70)
    print(f"Report saved to: {results_dir / 'test_report.json'}\n")

    return report


if __name__ == "__main__":
    res = run_comprehensive_tests()
    if res["verdict"] != "PASSED":
        sys.exit(1)

