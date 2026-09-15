"""Measure the actual DA-BALS admission path on a declared physical host.

This benchmark calls ``Gateway.admit`` directly. Request construction, random
number generation, file I/O, and summary statistics remain outside the timed
region. Results are suitable for the manuscript only when the author supplies
an identifiable physical device and controlled execution notes.
"""

from argparse import ArgumentParser
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import simpy

from run_experiments import Gateway, RequestSpec, WorkConservingCloud


POLICIES = (
    "DA-BALS-NoMargin",
    "DA-BALS-FixedMargin",
    "DA-BALS-Adaptive",
    "DA-BALS-Stabilized",
)
FROZEN = {"alpha": 0.05, "gamma": 0.75, "delta": 0.02, "m_max": 12.0}


def cpu_description():
    description = os.environ.get("PROCESSOR_IDENTIFIER", "")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                description = line.split(":", 1)[1].strip()
                break
    return description or platform.processor() or "unreported"


def bootstrap_median_ci(values, rng, resamples=10_000):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(resamples, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return np.quantile(medians, [0.025, 0.975])


def make_gateway(policy):
    env = simpy.Environment()
    cloud = WorkConservingCloud(env)
    gateway = Gateway(env, cloud, policy, 0, 1.2, 50.0, 1.0, [], tuning_params=FROZEN)
    gateway.visible_cloud_free_at = 8.0
    gateway.local_cloud_reserved_until = 6.0
    gateway.uplink_reserved_until = 4.0
    if policy == "DA-BALS-Adaptive":
        gateway.residuals = deque(np.linspace(0.25, 8.0, 20), maxlen=20)
    elif policy == "DA-BALS-Stabilized":
        gateway.stable_ewma = 4.0
        gateway.stable_margin = 3.0
    return gateway


def benchmark_policy(policy, requests, batches, warmup):
    samples = []
    for batch in range(warmup + batches):
        gateway = make_gateway(policy)
        start = time.perf_counter_ns()
        for request in requests:
            gateway.admit(request)
        elapsed = time.perf_counter_ns() - start
        if batch >= warmup:
            samples.append(elapsed / len(requests))
    return np.asarray(samples)


def main():
    parser = ArgumentParser()
    parser.add_argument("--device-label", required=True, help="Physical device model/name")
    parser.add_argument("--power-mode", required=True, help="Declared host power/performance mode")
    parser.add_argument("--notes", default="idle host; single Python process")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--decisions-per-batch", type=int, default=10_000)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    if args.batches < 30 or args.decisions_per_batch < 1_000 or args.warmup_batches < 5:
        raise ValueError("Publication run requires >=30 batches, >=1000 decisions/batch, and >=5 warm-up batches")

    rng = np.random.default_rng(20260914)
    sizes = rng.uniform(5.0, 15.0, args.decisions_per_batch)
    deadlines = rng.uniform(20_000.0, 40_000.0, args.decisions_per_batch)
    requests = [RequestSpec(i + 1, 0.0, float(sizes[i]), float(deadlines[i]), 0.5)
                for i in range(args.decisions_per_batch)]

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device_label": args.device_label,
        "cpu": cpu_description(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "power_mode": args.power_mode,
        "notes": args.notes,
        "timer": "time.perf_counter_ns",
        "batches": args.batches,
        "decisions_per_batch": args.decisions_per_batch,
        "warmup_batches": args.warmup_batches,
        "timed_scope": "Gateway.admit only; request generation and statistics excluded",
    }

    rows = []
    raw_rows = []
    bootstrap_rng = np.random.default_rng(20260915)
    for policy in POLICIES:
        values = benchmark_policy(policy, requests, args.batches, args.warmup_batches)
        low, high = bootstrap_median_ci(values, bootstrap_rng)
        rows.append({
            "policy": policy,
            "median_us": np.median(values) / 1_000.0,
            "p95_us": np.quantile(values, 0.95) / 1_000.0,
            "p99_us": np.quantile(values, 0.99) / 1_000.0,
            "median_ci95_low_us": low / 1_000.0,
            "median_ci95_high_us": high / 1_000.0,
            "batches": len(values),
            "decisions_per_batch": len(requests),
        })
        raw_rows.extend({"policy": policy, "batch": i + 1, "ns_per_decision": value}
                        for i, value in enumerate(values))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "hardware_decision_benchmark_summary.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(args.output_dir / "hardware_decision_benchmark_batches.csv", index=False)
    (args.output_dir / "hardware_decision_benchmark_environment.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
