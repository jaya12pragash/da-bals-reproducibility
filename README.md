# DA-BALS Reproducibility Repository

This repository contains the event-driven simulation, statistical analysis,
compact evidence, and figure-generation code for **Deadline-Aware Backpressure
and Stabilized Adaptive Load Shedding for Reliable Cloud--Edge AI Inference**.

DA-BALS evaluates request admission under uplink congestion, delayed cloud
telemetry, stochastic service, exogenous background work, and a mid-run
distribution shift. The protocol uses common random numbers and separate
development, holdout, independent robustness, ablation, and hybrid trace-replay
partitions.

## Verified primary result

At the predeclared peak-stress condition (`rho=1.2`, `tau=50 ms`, `CV=1.0`),
Stabilized achieved a mean on-time completion fraction of **68.748%** across 30
independent robustness seeds.

| Comparison | Mean difference | 95% paired CI |
|---|---:|---:|
| Stabilized - RollingQuantile | +3.191 pp | [1.819, 4.563] |
| Stabilized - FixedMargin | +1.066 pp | [0.768, 1.364] |
| Stabilized - NoMargin | -0.102 pp | [-0.302, 0.098] |

The last interval contains zero. The evidence supports correction of the
rolling-quantile degradation, not universal superiority over NoMargin.

## Verified extensions

- **Fresh-seed component ablation:** seeds 300--329 across all 54 operating
  conditions. At peak stress, none of the full-minus-reduced on-time intervals
  excluded zero after correction. The study therefore did not demonstrate that
  each stabilization component independently improved completion.
- **Hybrid Azure trace replay:** 30 non-overlapping blocks driven by production
  invocation timing and token-count ordering from Microsoft's Azure LLM
  inference trace. Stabilized exceeded RollingQuantile by 2.944 percentage
  points and FixedMargin by 0.694 points; no difference from NoMargin was
  detected. Deadlines, time scaling, service, and background traffic remain
  modeled, so this is a hybrid replay rather than production validation.
- **Hardware benchmark:** the reproducible benchmark program is supplied, but
  no hardware result is reported because no named physical-host evidence is
  part of this release.

## Repository structure

- `scripts/run_experiments.py` - event-driven simulator and shared workload code
- `scripts/tune_development_grid.py` - development-only parameter selection
- `scripts/run_holdout_evaluation.py` - untouched holdout evaluation
- `scripts/run_robustness_confirmation.py` - independent robustness confirmation
- `scripts/run_component_ablation.py` - fresh-seed stabilization ablation
- `scripts/run_hybrid_trace_replay.py` - prespecified hybrid trace replay
- `scripts/benchmark_decision_overhead.py` - physical-host timing harness
- `scripts/validate_model_consistency.py` - formula and information-boundary checks
- `scripts/validate_extended_claims.py` - ablation and trace-evidence checks
- `scripts/run_reference_validation.py` - 1,620-run CurrentState validation
- `scripts/generate_manuscript_figures.py` - numerical manuscript figures
- `scripts/generate_architecture_figure_v2.py` - vector architecture figure
- `results/` - frozen parameters and compact run-level/paired evidence
- `third_party/azure_llm_trace_2023/` - trace inputs, attribution, and license
- `figures/` - generated manuscript figures
- `docs/` - methodology, evidence manifest, and publication instructions

## Quick validation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r scripts/requirements.txt
python -m py_compile scripts/*.py
python scripts/validate_model_consistency.py
python scripts/validate_extended_claims.py
python scripts/generate_manuscript_figures.py
python scripts/generate_architecture_figure_v2.py
sha256sum -c SHA256SUMS
```

## Full experiment protocol

The parameter-selection boundary is essential. Do not retune after inspecting
holdout or robustness outputs.

```bash
python scripts/tune_development_grid.py
python scripts/run_holdout_evaluation.py
python scripts/run_robustness_confirmation.py
python scripts/apply_holm_corrections.py
python scripts/run_reference_validation.py
python scripts/run_component_ablation.py
python scripts/run_hybrid_trace_replay.py
```

The large request-level matrix is intentionally omitted. The committed compact
CSV files contain the run-level and paired statistics used for the reported
claims. The bundled Azure trace files retain their original CC BY 4.0 license
and attribution.

## Research boundaries

Telemetry becomes visible only after its configured delivery delay, and
completion residuals update adaptive state only after physical completion.
Successful-request p95 latency is conditional on successful completion and is
subject to survivorship effects. This repository does not establish physical
deployment performance, hardware throughput, universal optimality, or causal
contributions from every stabilization component.

## Citation and licenses

Citation metadata is provided in `CITATION.cff`. Update its publication fields
and DOI after formal publication. Source code is released under the MIT License.
Compact result tables and project documentation are released under CC BY 4.0;
see `DATA_LICENSE.txt`. Third-party trace licensing is recorded under
`third_party/azure_llm_trace_2023/`.
