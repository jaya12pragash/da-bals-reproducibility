# Reproducibility and Research Record

## Recommended GitHub workflow

1. Create a public repository named `da-bals-reproducibility`.
2. Upload or push the contents of this archive, not the enclosing directory.
3. Create the immutable Git tag `v1.1.0-submission`.
4. Create a GitHub Release from that tag and attach the repository ZIP.
5. Archive the release with Zenodo and add the assigned DOI to `CITATION.cff`.
6. After acceptance, create a new tagged release rather than rewriting the
   submission tag.

This history supplies dated evidence of the code, frozen parameters, compact
results, and exact validation state used for the manuscript.

Run both validation programs and verify `SHA256SUMS` immediately before tagging.
The hybrid replay can be regenerated from the bundled, attributed Azure trace
files. Hardware timing output must be generated on and reported with a named
physical host; this repository intentionally contains no such result.

## Boundaries

This repository supports reproducibility of the simulation study. It does not
constitute evidence of physical deployment, production line-rate performance,
or universal optimality. Those questions require separate experiments.
