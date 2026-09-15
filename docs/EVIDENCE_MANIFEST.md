# Evidence Manifest

| Evidence | Purpose |
|---|---|
| `results/frozen_parameters.json` | Records the development-selected controller parameters. |
| `results/development_grid_results.csv` | Records all 24 development-only tuning tuples. |
| `results/holdout_run_metrics.csv` | Seed-level holdout metrics across 54 conditions and four policies. |
| `results/holdout_paired_comparisons.csv` | Holdout paired differences and confidence intervals. |
| `results/robustness_run_metrics.csv` | Seed-level independent robustness metrics. |
| `results/robustness_paired_comparisons.csv` | Robustness paired tests and global 162-test Holm values. |
| `results/holm_nomargin_results.csv` | Matrix-wide Stabilized-versus-NoMargin correction results. |
| `results/current_state_validation_metrics.csv` | Complete 1,620-configuration validation of the corrected diagnostic reference. |
| `results/ablation_run_metrics.csv` | Fresh-seed component-ablation metrics for seeds 300--329. |
| `results/ablation_paired_comparisons.csv` | Full-minus-reduced paired intervals and adjusted tests. |
| `results/hybrid_trace_run_metrics.csv` | Run-level metrics for the 30-block hybrid Azure replay. |
| `results/hybrid_trace_paired_comparisons.csv` | Paired trace-replay comparisons and Holm-adjusted tests. |
| `results/extended_validation_protocol.json` | Frozen ablation and hybrid-replay protocol. |
| `third_party/azure_llm_trace_2023/` | Trace input files, source attribution, and CC BY 4.0 license. |

## Integrity policy

`SHA256SUMS` records hashes for the committed evidence and source files. When a
result is intentionally regenerated, update the hash manifest in the same
commit and explain the reason in `CHANGELOG.md`.

Do not replace compact evidence with manually edited values. Regenerate it using
the corresponding script and retain the console log or release artifact.
