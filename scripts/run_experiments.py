"""Leakage-free cloud-edge simulation for the DA-BALS evaluation.

All stochastic attributes are generated once per (seed, load) and replayed for
every policy and telemetry/CV configuration. Cloud telemetry consists only of
timestamped workload observations and becomes visible after ``tau``.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import simpy
from scipy.stats import norm

SEEDS = tuple(range(100, 130))
LOAD_POINTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
TAU_LEVELS = (0.0, 10.0, 50.0)
CV_LEVELS = (0.0, 0.5, 1.0)
POLICIES = (
    "FIFO", "STS", "DOA", "DA-BALS-NoFeedback", "DA-BALS-NoMargin",
    "DA-BALS-FixedMargin", "DA-BALS-Adaptive", "Perfect-Current-State",
)

GENERATION_HORIZON_MS = 5_000.0
SHIFT_EPOCH_MS = 2_500.0
LINK_RATE_UNITS_PER_MS = 1.0
BACKGROUND_RATE_PER_MS = 0.05
TELEMETRY_PERIOD_MS = 5.0
STS_THRESHOLD_UNITS = 30.0
CALIBRATION_WINDOW = 20
QUANTILE = 0.95
FIXED_MARGIN_MS = 4.0
STABILIZED_POLICIES = (
    "DA-BALS-EWMA",
    "DA-BALS-EWMA-Cap",
    "DA-BALS-EWMA-Decay",
    "DA-BALS-Stabilized",
)


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    birth_ms: float
    size_units: float
    relative_deadline_ms: float
    service_quantile: float

    @property
    def deadline_ms(self) -> float:
        return self.birth_ms + self.relative_deadline_ms


@dataclass(frozen=True)
class BackgroundSpec:
    job_id: int
    birth_ms: float
    service_quantile: float


def service_mean_for_birth(birth_ms: float) -> float:
    """The workload distribution shifts by arrival epoch, independent of policy."""
    return 4.0 if birth_ms >= SHIFT_EPOCH_MS else 2.0


def lognormal_from_quantile(mean_ms: float, cv: float, quantile: float) -> float:
    if cv == 0.0:
        return mean_ms
    sigma = np.sqrt(np.log1p(cv * cv))
    mu = np.log(mean_ms) - 0.5 * sigma * sigma
    q = np.clip(quantile, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return float(np.exp(mu + sigma * norm.ppf(q)))


def pregenerate_workload(seed: int, load: float) -> tuple[list[RequestSpec], list[BackgroundSpec]]:
    streams = np.random.SeedSequence([seed, int(round(load * 1000))]).spawn(5)
    arrival_rng, attribute_rng, fg_service_rng, bg_arrival_rng, bg_service_rng = (
        np.random.default_rng(s) for s in streams
    )
    requests: list[RequestSpec] = []
    t = 0.0
    rate = load / 10.0
    while True:
        t += float(arrival_rng.exponential(1.0 / rate))
        if t > GENERATION_HORIZON_MS:
            break
        requests.append(RequestSpec(
            len(requests) + 1, t, float(attribute_rng.uniform(5.0, 15.0)),
            float(attribute_rng.uniform(15.0, 60.0)), float(fg_service_rng.random()),
        ))
    background: list[BackgroundSpec] = []
    t = 0.0
    while True:
        t += float(bg_arrival_rng.exponential(1.0 / BACKGROUND_RATE_PER_MS))
        if t > GENERATION_HORIZON_MS:
            break
        background.append(BackgroundSpec(len(background) + 1, t, float(bg_service_rng.random())))
    return requests, background


class WorkConservingCloud:
    """Single-server FIFO cloud with exact active-plus-queued workload accounting."""

    def __init__(self, env: simpy.Environment):
        self.env = env
        self.resource = simpy.Resource(env, capacity=1)
        self._workload_ms = 0.0
        self._last_update_ms = 0.0

    def _advance(self) -> None:
        elapsed = self.env.now - self._last_update_ms
        self._workload_ms = max(0.0, self._workload_ms - elapsed)
        self._last_update_ms = self.env.now

    def observe_workload(self) -> float:
        self._advance()
        return self._workload_ms

    def enqueue(self, service_ms: float) -> None:
        self._advance()
        self._workload_ms += service_ms


class Gateway:
    def __init__(self, env, cloud, policy, seed, load, tau, cv, output,
                 state_output=None, tuning_params=None):
        self.env, self.cloud, self.policy = env, cloud, policy
        self.seed, self.load, self.tau, self.cv = seed, load, tau, cv
        self.output = output
        self.state_output = state_output
        self.state_rows = {}
        self.uplink = simpy.Resource(env, capacity=1)
        self.uplink_reserved_until = 0.0
        self.local_cloud_reserved_until = 0.0
        self.visible_cloud_free_at = 0.0
        self.latest_snapshot_epoch = -1.0
        self.residuals = deque(maxlen=CALIBRATION_WINDOW)
        params = tuning_params or {}
        self.stable_alpha = float(params.get("alpha", 0.15))
        self.stable_gamma = float(params.get("gamma", 1.50))
        self.stable_delta = float(params.get("delta", 0.02))
        self.stable_cap = float(params.get("m_max", 25.0))
        self.stable_ewma = 0.0
        self.stable_margin = 0.0
        self.stable_last_update = 0.0
        self.generated = 0
        self.terminal = 0

    def deliver_snapshot(self, observed_at: float, workload_ms: float):
        yield self.env.timeout(self.tau)
        if observed_at >= self.latest_snapshot_epoch:
            self.latest_snapshot_epoch = observed_at
            self.visible_cloud_free_at = observed_at + workload_ms

    def telemetry_monitor(self):
        while self.env.now <= GENERATION_HORIZON_MS:
            observed_at = self.env.now
            workload = self.cloud.observe_workload()
            self.env.process(self.deliver_snapshot(observed_at, workload))
            yield self.env.timeout(TELEMETRY_PERIOD_MS)

    def margin(self) -> float:
        if self.policy == "DA-BALS-FixedMargin":
            return FIXED_MARGIN_MS
        if self.policy == "DA-BALS-Adaptive" and len(self.residuals) >= 5:
            return max(0.0, float(np.quantile(tuple(self.residuals), QUANTILE)))
        if self.policy in STABILIZED_POLICIES:
            elapsed = max(0.0, self.env.now - self.stable_last_update)
            if self.policy in ("DA-BALS-EWMA-Decay", "DA-BALS-Stabilized"):
                self.stable_margin *= np.exp(-self.stable_delta * elapsed)
            self.stable_last_update = self.env.now
            return self.stable_margin
        return 0.0

    def admit(self, req: RequestSpec) -> tuple[bool, float]:
        now = self.env.now
        tx_finish = max(now, self.uplink_reserved_until) + req.size_units / LINK_RATE_UNITS_PER_MS
        mean_service = service_mean_for_birth(req.birth_ms)
        local_cloud_start = max(tx_finish, self.local_cloud_reserved_until)
        feedback_cloud_start = max(local_cloud_start, self.visible_cloud_free_at)
        active_margin = self.margin()
        if self.policy == "FIFO":
            predicted_finish, admitted = tx_finish + mean_service, True
        elif self.policy == "STS":
            predicted_finish = tx_finish + mean_service
            admitted = (self.uplink_reserved_until - now) * LINK_RATE_UNITS_PER_MS <= STS_THRESHOLD_UNITS
        elif self.policy == "DOA":
            predicted_finish = now + req.size_units / LINK_RATE_UNITS_PER_MS + mean_service
            admitted = predicted_finish <= req.deadline_ms
        elif self.policy == "DA-BALS-NoFeedback":
            predicted_finish = local_cloud_start + mean_service
            admitted = predicted_finish <= req.deadline_ms
        elif self.policy == "DA-BALS-NoMargin":
            predicted_finish = feedback_cloud_start + mean_service
            admitted = predicted_finish <= req.deadline_ms
        elif self.policy in ("DA-BALS-FixedMargin", "DA-BALS-Adaptive", *STABILIZED_POLICIES):
            predicted_finish = feedback_cloud_start + mean_service + active_margin
            admitted = predicted_finish <= req.deadline_ms
        elif self.policy == "Perfect-Current-State":
            exact_current_free = now + self.cloud.observe_workload()
            # This diagnostic policy observes exact current queued workload but
            # uses the same mean request-service estimate as the online policies.
            # The pregenerated realized service draw remains unavailable until
            # the request executes, preventing future-service information leakage.
            predicted_finish = max(tx_finish, exact_current_free) + mean_service
            admitted = predicted_finish <= req.deadline_ms
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        if admitted:
            self.uplink_reserved_until = tx_finish
            # Local advance reservation records this gateway's admitted work.
            reservation_start = feedback_cloud_start if self.policy.startswith("DA-BALS") else local_cloud_start
            self.local_cloud_reserved_until = reservation_start + mean_service
        if self.state_output is not None:
            residual_array = np.asarray(tuple(self.residuals), dtype=float)
            row = {
                "seed": self.seed, "load": self.load, "tau_ms": self.tau,
                "cv": self.cv, "policy": self.policy, "request_id": req.request_id,
                "arrival_ms": now, "deadline_ms": req.deadline_ms,
                "uplink_reserved_until_ms": self.uplink_reserved_until,
                "local_cloud_reserved_until_ms": self.local_cloud_reserved_until,
                "latest_snapshot_epoch_ms": self.latest_snapshot_epoch,
                "telemetry_age_ms": now - self.latest_snapshot_epoch if self.latest_snapshot_epoch >= 0 else np.nan,
                "visible_cloud_free_at_ms": self.visible_cloud_free_at,
                "actual_cloud_workload_at_admission_ms": self.cloud.observe_workload(),
                "margin_ms": active_margin, "residual_window_size": len(residual_array),
                "residual_mean_ms": residual_array.mean() if len(residual_array) else 0.0,
                "residual_p95_ms": np.quantile(residual_array, QUANTILE) if len(residual_array) else 0.0,
                "predicted_finish_ms": predicted_finish, "predicted_feasible": admitted,
                "realized_status": "Shed" if not admitted else "Pending",
                "realized_finish_ms": np.nan,
            }
            self.state_output.append(row)
            self.state_rows[req.request_id] = row
        return admitted, predicted_finish

    def route(self, req: RequestSpec):
        self.generated += 1
        admitted, predicted_finish = self.admit(req)
        if not admitted:
            self.record(req, "Shed", admitted=False, predicted_finish=predicted_finish)
            return
        with self.uplink.request() as token:
            yield token
            yield self.env.timeout(req.size_units / LINK_RATE_UNITS_PER_MS)
        cloud_arrival = self.env.now
        mean_service = service_mean_for_birth(req.birth_ms)
        service_ms = lognormal_from_quantile(mean_service, self.cv, req.service_quantile)
        self.cloud.enqueue(service_ms)
        with self.cloud.resource.request() as token:
            yield token
            cloud_start = self.env.now
            yield self.env.timeout(service_ms)
        finish = self.env.now
        status = "Success" if finish <= req.deadline_ms else "Late"
        self.residuals.append(max(0.0, finish - predicted_finish))
        if self.policy in STABILIZED_POLICIES:
            residual = max(0.0, finish - predicted_finish)
            self.stable_ewma = ((1.0-self.stable_alpha)*self.stable_ewma
                                + self.stable_alpha*residual)
            raw_margin = self.stable_gamma*self.stable_ewma
            if self.policy in ("DA-BALS-EWMA-Cap", "DA-BALS-Stabilized"):
                self.stable_margin = min(self.stable_cap, raw_margin)
            else:
                self.stable_margin = raw_margin
            self.stable_last_update = finish
        self.record(
            req, status, admitted=True, predicted_finish=predicted_finish,
            finish=finish, cloud_arrival=cloud_arrival, cloud_start=cloud_start,
            service_ms=service_ms,
        )

    def record(self, req, status, admitted, predicted_finish, finish=np.nan,
               cloud_arrival=np.nan, cloud_start=np.nan, service_ms=np.nan):
        late_network = bool(admitted and cloud_arrival > req.deadline_ms)
        post_deadline_compute = 0.0
        if admitted and np.isfinite(finish):
            post_deadline_compute = max(0.0, finish - max(cloud_start, req.deadline_ms))
        self.output.append({
            "seed": self.seed, "load": self.load, "tau_ms": self.tau, "cv": self.cv,
            "policy": self.policy, "request_id": req.request_id, "status": status,
            "admitted": admitted, "birth_ms": req.birth_ms, "deadline_ms": req.deadline_ms,
            "predicted_finish_ms": predicted_finish, "finish_ms": finish,
            "latency_ms": finish - req.birth_ms if np.isfinite(finish) else np.nan,
            "wasted_net_units": req.size_units if late_network else 0.0,
            "post_deadline_compute_ms": post_deadline_compute,
        })
        if self.state_output is not None and req.request_id in self.state_rows:
            state_row = self.state_rows[req.request_id]
            state_row["realized_status"] = status
            state_row["realized_finish_ms"] = finish
            state_row["realized_latency_ms"] = finish - req.birth_ms if np.isfinite(finish) else np.nan
        self.terminal += 1


def simulate(seed, load, tau, cv, policy, requests, background, output,
             state_output=None, tuning_params=None):
    env = simpy.Environment()
    cloud = WorkConservingCloud(env)
    gateway = Gateway(env, cloud, policy, seed, load, tau, cv, output,
                      state_output, tuning_params)

    def background_job(job: BackgroundSpec):
        yield env.timeout(job.birth_ms)
        service = lognormal_from_quantile(service_mean_for_birth(job.birth_ms), cv, job.service_quantile)
        cloud.enqueue(service)
        with cloud.resource.request() as token:
            yield token
            yield env.timeout(service)

    def foreground_generator():
        for req in requests:
            yield env.timeout(req.birth_ms - env.now)
            env.process(gateway.route(replace(req)))

    for job in background:
        env.process(background_job(job))
    env.process(foreground_generator())
    env.process(gateway.telemetry_monitor())
    env.run()
    assert gateway.generated == len(requests)
    assert gateway.terminal == len(requests)


def execute_matrix(output_path: Path, quick: bool = False) -> None:
    seeds = SEEDS[:2] if quick else SEEDS
    loads = LOAD_POINTS[:2] if quick else LOAD_POINTS
    rows: list[dict] = []
    total = len(seeds) * len(loads) * len(TAU_LEVELS) * len(CV_LEVELS) * len(POLICIES)
    completed = 0
    for seed in seeds:
        for load in loads:
            requests, background = pregenerate_workload(seed, load)
            for tau in TAU_LEVELS:
                for cv in CV_LEVELS:
                    for policy in POLICIES:
                        simulate(seed, load, tau, cv, policy, requests, background, rows)
                        completed += 1
                        if completed % 500 == 0 or completed == total:
                            print(f"completed {completed}/{total} configurations", flush=True)
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    expected = frame.groupby(["seed", "load", "tau_ms", "cv", "policy"]).size()
    baseline = frame.groupby(["seed", "load", "tau_ms", "cv", "policy"])["request_id"].nunique()
    assert expected.equals(baseline)
    print(f"wrote {len(frame):,} terminal request records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/journal_raw_experiments.csv"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    execute_matrix(args.output, args.quick)
