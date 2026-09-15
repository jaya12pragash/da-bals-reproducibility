"""Executable checks for manuscript-to-simulator consistency boundaries."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import RequestSpec, pregenerate_workload, simulate


def check_reference_has_no_realized_service_leak() -> None:
    source = Path(__file__).with_name("run_experiments.py").read_text()
    tree = ast.parse(source)
    gateway = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Gateway")
    admit = next(node for node in gateway.body if isinstance(node, ast.FunctionDef) and node.name == "admit")
    admit_source = ast.get_source_segment(source, admit)
    assert "req.service_quantile" not in admit_source
    assert "+ mean_service" in admit_source


def check_common_random_numbers() -> None:
    first, bg_first = pregenerate_workload(211, 1.2)
    second, bg_second = pregenerate_workload(211, 1.2)
    assert first == second
    assert bg_first == bg_second


def check_terminal_denominator_and_reference_sensitivity() -> None:
    requests, background = pregenerate_workload(211, 0.6)
    short_requests = requests[:80]
    short_background = [job for job in background if job.birth_ms <= short_requests[-1].birth_ms]
    baseline = []
    simulate(211, 0.6, 10.0, 0.5, "Perfect-Current-State",
             short_requests, short_background, baseline)
    frame = pd.DataFrame(baseline)
    assert len(frame) == len(short_requests)
    assert frame.request_id.nunique() == len(short_requests)
    assert set(frame.status).issubset({"Success", "Late", "Shed"})

    # Alter only the first request's realized service quantile in an otherwise
    # empty system. Its admission decision must remain identical because the
    # reference uses mean service, not that future draw. Later decisions are
    # intentionally excluded: changed realized work can legitimately change
    # the current queue state observed by those arrivals.
    original = [short_requests[0]]
    changed = [RequestSpec(original[0].request_id, original[0].birth_ms,
                           original[0].size_units, original[0].relative_deadline_ms,
                           1.0-original[0].service_quantile)]
    original_rows, changed_rows = [], []
    simulate(211, 0.6, 10.0, 0.5, "Perfect-Current-State",
             original, [], original_rows)
    simulate(211, 0.6, 10.0, 0.5, "Perfect-Current-State",
             changed, [], changed_rows)
    assert original_rows[0]["admitted"] == changed_rows[0]["admitted"]


def main() -> None:
    check_reference_has_no_realized_service_leak()
    check_common_random_numbers()
    check_terminal_denominator_and_reference_sensitivity()
    print("PASS: formula/state boundary, CRN, terminal denominator, and reference-information checks")


if __name__ == "__main__":
    main()
