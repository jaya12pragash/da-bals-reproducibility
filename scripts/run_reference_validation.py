"""Validate the leakage-free CurrentState diagnostic over the full 30-seed matrix."""

from pathlib import Path

import pandas as pd

from run_experiments import CV_LEVELS, LOAD_POINTS, SEEDS, TAU_LEVELS, pregenerate_workload, simulate

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    summaries = []
    total = len(SEEDS) * len(LOAD_POINTS) * len(TAU_LEVELS) * len(CV_LEVELS)
    completed = 0
    for seed in SEEDS:
        for load in LOAD_POINTS:
            requests, background = pregenerate_workload(seed, load)
            for tau in TAU_LEVELS:
                for cv in CV_LEVELS:
                    rows = []
                    simulate(seed, load, tau, cv, "Perfect-Current-State",
                             requests, background, rows)
                    frame = pd.DataFrame(rows)
                    assert len(frame) == len(requests)
                    assert frame.request_id.nunique() == len(requests)
                    summaries.append({
                        "seed": seed,
                        "load": load,
                        "tau_ms": tau,
                        "cv": cv,
                        "requests": len(frame),
                        "on_time_fraction_pct": 100.0 * frame.status.eq("Success").mean(),
                        "late_fraction_pct": 100.0 * frame.status.eq("Late").mean(),
                        "shed_fraction_pct": 100.0 * frame.status.eq("Shed").mean(),
                    })
                    completed += 1
                    if completed % 250 == 0 or completed == total:
                        print(f"completed {completed}/{total} CurrentState configurations", flush=True)

    output = ROOT / "results" / "current_state_validation_metrics.csv"
    pd.DataFrame(summaries).to_csv(output, index=False)
    print(f"PASS: wrote {len(summaries)} complete run summaries to {output}")


if __name__ == "__main__":
    main()
