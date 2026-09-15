# Publish This Repository to GitHub

## Recommended repository settings

- Repository name: `da-bals-reproducibility`
- Visibility: Public
- Description: `Reproducible simulation and statistical evidence for stabilized deadline-aware cloud-edge admission control.`
- Topics: `edge-computing`, `cloud-computing`, `load-shedding`, `backpressure`, `simpy`, `reproducible-research`, `adaptive-control`

Do not ask GitHub to create a README, license, or `.gitignore`; this package
already includes them.

## PowerShell commands

Extract the ZIP, open PowerShell inside the extracted directory, and run:

```powershell
git init
git branch -M main
git add .
git commit -m "Release DA-BALS reproducibility package v1.1.0"
git remote add origin https://github.com/jaya12pragash/da-bals-reproducibility.git
git push -u origin main
git tag -a v1.1.0-submission -m "IEEE Access submission evidence"
git push origin v1.1.0-submission
```

Create a GitHub Release from `v1.1.0-submission`. Keep that tag immutable. For
later corrections, create `v1.1.1` or a newer version rather than moving or
deleting the submission tag.

## Evidence preservation

GitHub commit history establishes when files were published, but it is not a
formal archival DOI. After creating the release, connect the repository to
Zenodo and archive the tagged release. Add the resulting DOI to `CITATION.cff`
in a subsequent version.
