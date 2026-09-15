"""Hybrid replay of the Microsoft Azure LLM inference trace 2023.

The protocol is frozen in results/extended_validation_protocol.json. Real
timestamp spacings and token-count order statistics drive the foreground
requests. Deadline, service calibration, and background work remain modeled.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats

from run_experiments import pregenerate_workload, RequestSpec, simulate
from run_holdout_evaluation import holm_adjust


REPLICATES = 30
REQUESTS_PER_REPLICATE = 600
TRACE_SEEDS = tuple(range(500, 530))
POLICIES = (
    "DA-BALS-NoMargin",
    "DA-BALS-FixedMargin",
    "DA-BALS-Adaptive",
    "DA-BALS-Stabilized",
)
LOAD = 1.2
TAU_MS = 50.0
CV = 1.0
HORIZON_MS = 5_000.0


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(trace_dir):
    paths = [trace_dir / "AzureLLMInferenceTrace_code.csv", trace_dir / "AzureLLMInferenceTrace_conv.csv"]
    frames = []
    for source_rank, path in enumerate(paths):
        frame = pd.read_csv(path)
        assert list(frame.columns) == ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
        frame["timestamp"] = pd.to_datetime(frame.TIMESTAMP)
        frame["source_rank"] = source_rank
        frame["source_row"] = np.arange(len(frame))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "source_rank", "source_row"], kind="mergesort"
    ).reset_index(drop=True)
    needed = REPLICATES * REQUESTS_PER_REPLICATE
    assert len(combined) >= needed
    return combined.iloc[:needed].copy(), {p.name: sha256(p) for p in paths}


def rank_to_interval(series, low, high):
    ranks = series.rank(method="average").to_numpy(dtype=float)
    return low + (high - low) * (ranks - 0.5) / len(series)


def build_requests(block, seed):
    elapsed = (block.timestamp - block.timestamp.iloc[0]).dt.total_seconds().to_numpy() * 1000.0
    span = elapsed[-1]
    assert span > 0.0
    births = 0.001 + elapsed / span * (HORIZON_MS - 0.002)
    sizes = rank_to_interval(block.ContextTokens, 5.0, 15.0)
    service_quantiles = rank_to_interval(block.GeneratedTokens, 0.0, 1.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2023, 600]))
    deadlines = rng.uniform(15.0, 60.0, len(block))
    requests = [
        RequestSpec(i + 1, float(births[i]), float(sizes[i]), float(deadlines[i]), float(service_quantiles[i]))
        for i in range(len(block))
    ]
    assert abs(np.mean([r.size_units for r in requests]) - 10.0) < 1e-12
    return requests


def summarize(outcomes, requests):
    frame = pd.DataFrame(outcomes)
    assert len(frame) == len(requests)
    offered = sum(r.size_units for r in requests)
    return {
        "requests": len(frame),
        "on_time_fraction_pct": 100.0 * frame.status.eq("Success").mean(),
        "late_fraction_pct": 100.0 * frame.status.eq("Late").mean(),
        "shed_fraction_pct": 100.0 * frame.status.eq("Shed").mean(),
        "normalized_network_waste": frame.wasted_net_units.sum() / offered,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "results" / "extended_validation_protocol.json").read_text())
    assert protocol["frozen_before_execution"] is True
    trace, hashes = load_trace(root / "third_party" / "azure_llm_trace_2023")
    frozen = json.loads((root / "results" / "frozen_parameters.json").read_text())["selected_parameters"]

    rows = []
    for replicate, seed in enumerate(TRACE_SEEDS):
        start = replicate * REQUESTS_PER_REPLICATE
        block = trace.iloc[start:start + REQUESTS_PER_REPLICATE].reset_index(drop=True)
        requests = build_requests(block, seed)
        _, background = pregenerate_workload(seed, LOAD)
        for policy in POLICIES:
            outcomes = []
            simulate(seed, LOAD, TAU_MS, CV, policy, requests, background, outcomes,
                     tuning_params=frozen if policy == "DA-BALS-Stabilized" else None)
            rows.append({
                "replicate": replicate + 1,
                "seed": seed,
                "policy": policy,
                "trace_start": block.TIMESTAMP.iloc[0],
                "trace_end": block.TIMESTAMP.iloc[-1],
                **summarize(outcomes, requests),
            })

    runs = pd.DataFrame(rows)
    output = root / "results"
    runs.to_csv(output / "hybrid_trace_run_metrics.csv", index=False)

    full = runs[runs.policy == "DA-BALS-Stabilized"][["replicate", "on_time_fraction_pct"]]
    comparisons = []
    for baseline in POLICIES[:-1]:
        base = runs[runs.policy == baseline][["replicate", "on_time_fraction_pct"]]
        paired = full.merge(base, on="replicate", suffixes=("_stabilized", "_baseline"), validate="one_to_one")
        delta = paired.on_time_fraction_pct_stabilized - paired.on_time_fraction_pct_baseline
        mean = float(delta.mean())
        half = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
        p_value = 1.0 if np.allclose(delta, 0.0) else float(stats.ttest_1samp(delta, 0.0).pvalue)
        comparisons.append({
            "baseline": baseline,
            "pairs": len(delta),
            "mean_delta_pp": mean,
            "ci95_low_pp": mean - half,
            "ci95_high_pp": mean + half,
            "p_value": p_value,
        })
    paired = pd.DataFrame(comparisons)
    paired["holm_p_value_3"] = holm_adjust(paired.p_value)
    paired.to_csv(output / "hybrid_trace_paired_comparisons.csv", index=False)

    means = runs.groupby("policy").agg(
        on_time_mean_pct=("on_time_fraction_pct", "mean"),
        late_mean_pct=("late_fraction_pct", "mean"),
        shed_mean_pct=("shed_fraction_pct", "mean"),
    )
    report = (
        "TRACE INPUT SHA-256\n" + json.dumps(hashes, indent=2)
        + "\n\nHYBRID TRACE REPLAY MEANS\n" + means.to_string()
        + "\n\nPAIRED STABILIZED-MINUS-BASELINE COMPARISONS\n" + paired.to_string(index=False) + "\n"
    )
    (output / "hybrid_trace_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
