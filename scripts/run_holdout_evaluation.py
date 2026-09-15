"""One-time evaluation on untouched holdout seeds 115--129."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

from run_experiments import LOAD_POINTS, TAU_LEVELS, CV_LEVELS, pregenerate_workload, simulate

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_SEEDS = tuple(range(115, 130))
POLICIES = ("DA-BALS-NoMargin", "DA-BALS-FixedMargin",
            "DA-BALS-Adaptive", "DA-BALS-Stabilized")
MANIFEST = ROOT / "results" / "frozen_parameters.json"


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m-rank)*values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["holdout_accessed"] is False
    frozen = manifest["selected_parameters"]
    run_rows = []
    total = len(HOLDOUT_SEEDS)*len(LOAD_POINTS)*len(TAU_LEVELS)*len(CV_LEVELS)*len(POLICIES)
    completed = 0
    for seed in HOLDOUT_SEEDS:
        for load in LOAD_POINTS:
            requests, background = pregenerate_workload(seed, load)
            offered_units = sum(r.size_units for r in requests)
            for tau in TAU_LEVELS:
                for cv in CV_LEVELS:
                    for policy in POLICIES:
                        outcomes = []
                        params = frozen if policy == "DA-BALS-Stabilized" else None
                        simulate(seed, load, tau, cv, policy, requests, background,
                                 outcomes, tuning_params=params)
                        frame = pd.DataFrame(outcomes)
                        assert len(frame) == len(requests)
                        run_rows.append({
                            "seed": seed, "load": load, "tau_ms": tau, "cv": cv,
                            "policy": policy, "requests": len(frame),
                            "on_time_fraction_pct": 100*frame.status.eq("Success").mean(),
                            "late_fraction_pct": 100*frame.status.eq("Late").mean(),
                            "shed_fraction_pct": 100*frame.status.eq("Shed").mean(),
                            "normalized_network_waste": frame.wasted_net_units.sum()/offered_units,
                        })
                        completed += 1
                        if completed % 250 == 0 or completed == total:
                            print(f"completed {completed}/{total} holdout configurations", flush=True)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    runs = pd.DataFrame(run_rows)
    runs.to_csv(output/"holdout_run_metrics.csv", index=False)

    comparisons = []
    for load in LOAD_POINTS:
        for tau in TAU_LEVELS:
            for cv in CV_LEVELS:
                block = runs[(runs.load==load)&(runs.tau_ms==tau)&(runs.cv==cv)]
                target = block[block.policy=="DA-BALS-Stabilized"][["seed","on_time_fraction_pct"]]
                for baseline in POLICIES[:-1]:
                    base = block[block.policy==baseline][["seed","on_time_fraction_pct"]]
                    paired = target.merge(base, on="seed", suffixes=("_stabilized","_baseline"), validate="one_to_one")
                    assert len(paired)==15
                    delta = paired.on_time_fraction_pct_stabilized-paired.on_time_fraction_pct_baseline
                    mean = delta.mean()
                    half = stats.t.ppf(.975,14)*stats.sem(delta)
                    test = stats.ttest_1samp(delta,0.0)
                    comparisons.append({"load":load,"tau_ms":tau,"cv":cv,"baseline":baseline,
                        "pairs":15,"mean_delta_pp":mean,"ci95_low_pp":mean-half,
                        "ci95_high_pp":mean+half,"p_value":test.pvalue})
    paired = pd.DataFrame(comparisons)
    paired["holm_p_value_global_exploratory"] = holm_adjust(paired.p_value)
    paired.to_csv(output/"holdout_paired_comparisons.csv", index=False)

    stress = paired[(paired.load==1.2)&(paired.tau_ms==50)&(paired.cv==1.0)]
    stress.to_csv(output/"holdout_primary_stress_comparisons.csv", index=False)
    means = runs[(runs.load==1.2)&(runs.tau_ms==50)&(runs.cv==1.0)].groupby("policy").on_time_fraction_pct.agg(["mean","std"])
    with (output/"holdout_report.txt").open("w") as report:
        report.write("FROZEN PARAMETERS\n"+json.dumps(frozen,indent=2)+"\n\n")
        report.write("PRIMARY STRESS POLICY MEANS\n"+means.to_string()+"\n\n")
        report.write("PRIMARY STRESS PAIRED COMPARISONS\n"+stress.to_string(index=False)+"\n")
    print((output/"holdout_report.txt").read_text())


if __name__ == "__main__":
    main()
