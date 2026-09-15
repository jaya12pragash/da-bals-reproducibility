"""Aggregate runs and calculate predeclared paired confidence intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "journal_raw_experiments.csv"
OUT = ROOT / "results"
df = pd.read_csv(RAW)

keys = ["seed", "load", "tau_ms", "cv", "policy"]
records = []
for values, group in df.groupby(keys, sort=True):
    success = group.status.eq("Success")
    admitted = group.admitted.astype(bool)
    successful_latency = group.loc[success, "latency_ms"]
    records.append(dict(zip(keys, values)) | {
        "requests": len(group),
        "on_time_fraction_pct": 100.0 * success.mean(),
        "admission_fraction_pct": 100.0 * admitted.mean(),
        "false_admission_pct": 100.0 * (group.status.eq("Late") & admitted).sum() / max(1, admitted.sum()),
        "conditional_p95_latency_ms": successful_latency.quantile(0.95) if len(successful_latency) else np.nan,
        "wasted_net_units": group.wasted_net_units.sum(),
        "post_deadline_compute_ms": group.post_deadline_compute_ms.sum(),
    })
summary = pd.DataFrame(records)
OUT.mkdir(exist_ok=True)
summary.to_csv(OUT / "run_level_summary.csv", index=False)

comparisons = []
for load in sorted(summary.load.unique()):
    for tau in sorted(summary.tau_ms.unique()):
        for cv in sorted(summary.cv.unique()):
            block = summary[(summary.load == load) & (summary.tau_ms == tau) & (summary.cv == cv)]
            adaptive = block[block.policy == "DA-BALS-Adaptive"][["seed", "on_time_fraction_pct"]]
            for baseline in ("DA-BALS-NoMargin", "DA-BALS-FixedMargin"):
                other = block[block.policy == baseline][["seed", "on_time_fraction_pct"]]
                paired = adaptive.merge(other, on="seed", suffixes=("_adaptive", "_baseline"), validate="one_to_one")
                assert len(paired) == summary.seed.nunique()
                delta = paired.on_time_fraction_pct_adaptive - paired.on_time_fraction_pct_baseline
                mean = delta.mean()
                half = stats.t.ppf(0.975, len(delta)-1) * stats.sem(delta)
                comparisons.append({
                    "load": load, "tau_ms": tau, "cv": cv, "baseline": baseline,
                    "pairs": len(delta), "mean_delta_pp": mean,
                    "ci95_low_pp": mean-half, "ci95_high_pp": mean+half,
                    "favors_adaptive": mean-half > 0,
                })
paired = pd.DataFrame(comparisons)
paired.to_csv(OUT / "paired_adaptive_comparisons.csv", index=False)
print(paired.to_string(index=False))
