"""Predeclared development-only tuning for DA-BALS-Stabilized.

Holdout seeds 115--129 are never loaded by this program.
"""
from itertools import product
from pathlib import Path
import json
import pandas as pd

from run_experiments import pregenerate_workload, simulate

ROOT = Path(__file__).resolve().parents[1]

DEV_SEEDS = tuple(range(100, 115))
DEV_LOADS = (0.8, 1.0, 1.2)
DEV_TAUS = (10.0, 50.0)
DEV_CVS = (0.5, 1.0)
PARAM_GRID = [
    {"alpha": a, "gamma": g, "delta": d, "m_max": cap}
    for a, g, d, cap in product(
        (0.05, 0.15, 0.25), (0.75, 1.25), (0.005, 0.02), (12.0, 24.0)
    )
]
LATE_WEIGHT = 0.50
SHED_WEIGHT = 0.10
WASTE_WEIGHT = 0.02


def evaluate(params):
    rows = []
    offered_units = 0.0
    offered_requests = 0
    for seed in DEV_SEEDS:
        for load in DEV_LOADS:
            requests, background = pregenerate_workload(seed, load)
            for tau in DEV_TAUS:
                for cv in DEV_CVS:
                    before = len(rows)
                    simulate(seed, load, tau, cv, "DA-BALS-Stabilized",
                             requests, background, rows, tuning_params=params)
                    assert len(rows)-before == len(requests)
                    offered_units += sum(r.size_units for r in requests)
                    offered_requests += len(requests)
    df = pd.DataFrame(rows)
    success = df.status.eq("Success").mean()
    late = df.status.eq("Late").mean()
    shed = df.status.eq("Shed").mean()
    normalized_waste = df.wasted_net_units.sum()/offered_units
    score = success-LATE_WEIGHT*late-SHED_WEIGHT*shed-WASTE_WEIGHT*normalized_waste
    return {**params, "J": score, "success_fraction": success,
            "late_fraction": late, "shed_fraction": shed,
            "normalized_network_waste": normalized_waste,
            "records": offered_requests}


def main():
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    results = []
    for index, params in enumerate(PARAM_GRID, 1):
        result = evaluate(params)
        results.append(result)
        print(f"{index:02d}/{len(PARAM_GRID)} J={result['J']:.6f} {params}", flush=True)
    frame = pd.DataFrame(results).sort_values(
        ["J", "success_fraction", "late_fraction"], ascending=[False, False, True]
    )
    frame.to_csv(output/"development_grid_results.csv", index=False)
    best = {key: float(frame.iloc[0][key]) for key in ("alpha", "gamma", "delta", "m_max")}
    manifest = {
        "selected_parameters": best,
        "selection_objective": "success - 0.50*late - 0.10*shed - 0.02*normalized_network_waste",
        "development_seeds": [100, 114],
        "development_conditions": {"loads": DEV_LOADS, "tau_ms": DEV_TAUS, "cv": DEV_CVS},
        "holdout_accessed": False,
    }
    (output/"frozen_parameters.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("\nSELECTED AND FROZEN")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
