# Contributing

Changes that affect experimental results must include:

1. A description of the model or analysis change.
2. Regenerated compact evidence from the applicable script.
3. Updated figures when their input data changed.
4. Updated `SHA256SUMS` and `CHANGELOG.md`.
5. Passing automated consistency checks.

Do not modify frozen parameters in a holdout or robustness release. A new
parameter search requires a new development protocol and a separately versioned
experiment.
