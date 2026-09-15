"""Predeclared component ablation on fresh seeds 300--329.

The frozen full controller is compared with three reduced variants across the
same 54-condition matrix. Common random numbers are retained within every
seed/condition block. This program writes only run-level metrics.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

from run_experiments import (
    CV_LEVELS,
    LOAD_POINTS,
    TAU_LEVELS,
    pregenerate_workload,
    simulate,
)
from run_holdout_evaluation import holm_adjust


ABLATION_SEEDS = tuple(range(300, 330))
POLICIES = (
    "DA-BALS-EWMA",
    "DA-BALS-EWMA-Cap",
    "DA-BALS-EWMA-Decay",
    "DA-BALS-Stabilized",
)
PRIMARY = (1.2, 50.0, 1.0)


def summarize(outcomes, requests, offered_units):
    frame = pd.DataFrame(outcomes)
    assert len(frame) == len(requests)
    return {
        "requests": len(frame),
        "on_time_fraction_pct": 100.0 * frame.status.eq("Success").mean(),
        "late_fraction_pct": 100.0 * frame.status.eq("Late").mean(),
        "shed_fraction_pct": 100.0 * frame.status.eq("Shed").mean(),
        "normalized_network_waste": frame.wasted_net_units.sum() / offered_units,
    }


def paired_comparisons(runs):
    rows = []
    for load in LOAD_POINTS:
        for tau in TAU_LEVELS:
            for cv in CV_LEVELS:
                block = runs[(runs.load == load) & (runs.tau_ms == tau) & (runs.cv == cv)]
                full = block[block.policy == "DA-BALS-Stabilized"][["seed", "on_time_fraction_pct"]]
                for reduced in POLICIES[:-1]:
                    base = block[block.policy == reduced][["seed", "on_time_fraction_pct"]]
                    paired = full.merge(base, on="seed", suffixes=("_full", "_reduced"), validate="one_to_one")
                    assert len(paired) == len(ABLATION_SEEDS)
                    delta = paired.on_time_fraction_pct_full - paired.on_time_fraction_pct_reduced
                    mean = float(delta.mean())
                    half = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
                    p_value = 1.0 if np.allclose(delta, 0.0) else float(stats.ttest_1samp(delta, 0.0).pvalue)
                    rows.append({
                        "load": load,
                        "tau_ms": tau,
                        "cv": cv,
                        "reduced_variant": reduced,
                        "pairs": len(delta),
                        "mean_delta_pp": mean,
                        "ci95_low_pp": mean - half,
                        "ci95_high_pp": mean + half,
                        "p_value": p_value,
                    })
    result = pd.DataFrame(rows)
    result["holm_p_value_global_162"] = holm_adjust(result.p_value)
    is_primary = (
        (result.load == PRIMARY[0])
        & (result.tau_ms == PRIMARY[1])
        & (result.cv == PRIMARY[2])
    )
    result["holm_p_value_primary_3"] = np.nan
    result.loc[is_primary, "holm_p_value_primary_3"] = holm_adjust(result.loc[is_primary, "p_value"])
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "results" / "extended_validation_protocol.json").read_text())
    assert protocol["frozen_before_execution"] is True
    assert protocol["component_ablation"]["seeds"] == [300, 329]
    frozen = json.loads((root / "results" / "frozen_parameters.json").read_text())["selected_parameters"]

    rows = []
    total = len(ABLATION_SEEDS) * len(LOAD_POINTS) * len(TAU_LEVELS) * len(CV_LEVELS) * len(POLICIES)
    completed = 0
    for seed in ABLATION_SEEDS:
        for load in LOAD_POINTS:
            requests, background = pregenerate_workload(seed, load)
            offered_units = sum(r.size_units for r in requests)
            for tau in TAU_LEVELS:
                for cv in CV_LEVELS:
                    for policy in POLICIES:
                        outcomes = []
                        simulate(seed, load, tau, cv, policy, requests, background, outcomes, tuning_params=frozen)
                        rows.append({
                            "seed": seed,
                            "load": load,
                            "tau_ms": tau,
                            "cv": cv,
                            "policy": policy,
                            **summarize(outcomes, requests, offered_units),
                        })
                        completed += 1
                        if completed % 500 == 0 or completed == total:
                            print(f"completed {completed}/{total} ablation configurations", flush=True)

    runs = pd.DataFrame(rows)
    output = root / "results"
    runs.to_csv(output / "ablation_run_metrics.csv", index=False)
    paired = paired_comparisons(runs)
    paired.to_csv(output / "ablation_paired_comparisons.csv", index=False)

    primary = paired[(paired.load == PRIMARY[0]) & (paired.tau_ms == PRIMARY[1]) & (paired.cv == PRIMARY[2])]
    means = runs[(runs.load == PRIMARY[0]) & (runs.tau_ms == PRIMARY[1]) & (runs.cv == PRIMARY[2])].groupby("policy").agg(
        on_time_mean_pct=("on_time_fraction_pct", "mean"),
        late_mean_pct=("late_fraction_pct", "mean"),
        shed_mean_pct=("shed_fraction_pct", "mean"),
    )
    report = "PRIMARY STRESS ABLATION MEANS\n" + means.to_string() + "\n\nPAIRED FULL-MINUS-REDUCED COMPARISONS\n" + primary.to_string(index=False) + "\n"
    (output / "ablation_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
