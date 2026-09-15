"""Validate manuscript-level ablation and hybrid trace claims."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = (1.2, 50.0, 1.0)


def close(actual: float, expected: float, atol: float = 5e-4) -> None:
    assert np.isclose(actual, expected, atol=atol), (actual, expected)


def validate_ablation() -> None:
    runs = pd.read_csv(ROOT / "results" / "ablation_run_metrics.csv")
    paired = pd.read_csv(ROOT / "results" / "ablation_paired_comparisons.csv")
    assert len(runs) == 30 * 54 * 4
    assert set(runs.seed) == set(range(300, 330))
    block = runs[(runs.load == PRIMARY[0]) & (runs.tau_ms == PRIMARY[1]) & (runs.cv == PRIMARY[2])]
    expected = {
        "DA-BALS-EWMA": (69.652110, 3.752104, 26.595786),
        "DA-BALS-EWMA-Cap": (69.652110, 3.757395, 26.590495),
        "DA-BALS-EWMA-Decay": (69.624284, 3.882014, 26.493702),
        "DA-BALS-Stabilized": (69.640694, 3.887305, 26.472001),
    }
    for policy, values in expected.items():
        row = block[block.policy == policy]
        close(row.on_time_fraction_pct.mean(), values[0])
        close(row.late_fraction_pct.mean(), values[1])
        close(row.shed_fraction_pct.mean(), values[2])
    primary = paired[(paired.load == PRIMARY[0]) & (paired.tau_ms == PRIMARY[1]) & (paired.cv == PRIMARY[2])]
    assert len(primary) == 3
    assert (primary.ci95_low_pp < 0).all() and (primary.ci95_high_pp > 0).all()
    assert (primary.holm_p_value_primary_3 >= 0.05).all()


def validate_trace() -> None:
    runs = pd.read_csv(ROOT / "results" / "hybrid_trace_run_metrics.csv")
    paired = pd.read_csv(ROOT / "results" / "hybrid_trace_paired_comparisons.csv")
    assert len(runs) == 30 * 4
    assert set(runs.seed) == set(range(500, 530))
    expected_means = {
        "DA-BALS-Adaptive": 60.405556,
        "DA-BALS-FixedMargin": 62.655556,
        "DA-BALS-NoMargin": 63.238889,
        "DA-BALS-Stabilized": 63.350000,
    }
    for policy, value in expected_means.items():
        close(runs.loc[runs.policy == policy, "on_time_fraction_pct"].mean(), value)
    expected_deltas = {
        "DA-BALS-Adaptive": (2.944444, 2.086551, 3.802338, True),
        "DA-BALS-FixedMargin": (0.694444, 0.450885, 0.938003, True),
        "DA-BALS-NoMargin": (0.111111, -0.077608, 0.299830, False),
    }
    for baseline, values in expected_deltas.items():
        row = paired[paired.baseline == baseline].iloc[0]
        close(row.mean_delta_pp, values[0])
        close(row.ci95_low_pp, values[1])
        close(row.ci95_high_pp, values[2])
        assert bool(row.holm_p_value_3 < 0.05) is values[3]


if __name__ == "__main__":
    validate_ablation()
    validate_trace()
    print("PASS: ablation and hybrid trace claims match compact evidence")
