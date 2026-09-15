# Experimental Methodology

## Workload

- Foreground arrivals: Poisson process over a 5,000 ms generation horizon.
- Request size: Uniform(5, 15) workload units.
- Relative deadline: Uniform(15, 60) ms.
- Cloud service: lognormal with `CV` in `{0, 0.5, 1.0}`.
- Workload shift: mean cloud service changes from 2 ms to 4 ms at 2,500 ms.
- Background traffic: independent Poisson arrivals at 0.05 jobs/ms.
- Uplink: one work-conserving server at one unit/ms.
- Cloud: one work-conserving FIFO processor.

Arrivals stop at the generation horizon, but the event environment drains all
scheduled foreground and background work. Every generated foreground request
must produce exactly one terminal status: `Success`, `Late`, or `Shed`.

## Experimental matrix

- Loads: `{0.2, 0.4, 0.6, 0.8, 1.0, 1.2}`
- Telemetry delays: `{0, 10, 50}` ms
- Service variability: `{0, 0.5, 1.0}`
- Development seeds: `100--114`
- Holdout seeds: `115--129`
- Robustness seeds: `200--229`
- Component-ablation seeds: `300--329`
- Hybrid trace-replay seeds: `500--529`

## Frozen controller

The development search evaluated 24 tuples over 12 difficult conditions. The
selected parameters were frozen before holdout evaluation:

- `alpha = 0.05`
- `gamma = 0.75`
- `delta = 0.02 /ms`
- `m_max = 12 ms`

The selection objective was

`J = success - 0.50*late - 0.10*shed - 0.02*normalized_network_waste`.

## Statistical analysis

Comparisons are paired by seed and operating condition. Two-sided 95%
Student-t intervals are calculated on seed-level differences. Holm-Bonferroni
adjustment controls family-wise error across 54 conditions for each comparison;
a secondary global family contains 162 tests.

Absence of statistical significance is reported as "no detected difference,"
not equivalence. Successful-request p95 latency is conditional on success and
is not interpreted as an unconditional latency guarantee.

## Extension protocols

The fresh-seed ablation held the frozen controller parameters fixed and
evaluated EWMA-only, EWMA-plus-cap, EWMA-plus-decay, and full Stabilized variants
over the complete 54-condition matrix. The hybrid replay used 30 non-overlapping
600-request blocks from the Azure LLM inference trace. Trace timestamps and
token-count orderings were empirical; deadlines, time scaling, service demand,
background work, and the cloud-edge topology remained modeled.

These extensions do not establish that each stabilization component has an
independent causal effect, and the hybrid replay is not a deployment study.
