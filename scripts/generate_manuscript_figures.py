from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS / "robustness_run_metrics.csv")
policies = ["DA-BALS-NoMargin", "DA-BALS-FixedMargin", "DA-BALS-Adaptive", "DA-BALS-Stabilized"]
labels = {"DA-BALS-NoMargin": "No margin", "DA-BALS-FixedMargin": "Fixed margin",
          "DA-BALS-Adaptive": "Rolling quantile", "DA-BALS-Stabilized": "Stabilized adaptive"}
colors = {"DA-BALS-NoMargin": "#595959", "DA-BALS-FixedMargin": "#e68613",
          "DA-BALS-Adaptive": "#a23bec", "DA-BALS-Stabilized": "#1769aa"}
markers = {"DA-BALS-NoMargin": "o", "DA-BALS-FixedMargin": "^",
           "DA-BALS-Adaptive": "x", "DA-BALS-Stabilized": "s"}

def mean_ci(values):
    x = np.asarray(values, dtype=float)
    m = np.mean(x)
    h = stats.t.ppf(.975, len(x)-1) * stats.sem(x) if len(x) > 1 else 0
    return m, h

# Fig. 1: load sweep at the predeclared nominal uncertainty slice.
nom = df[(df.tau_ms == 10.0) & (df.cv == 0.5)]
fig, ax = plt.subplots(figsize=(7.2, 4.25))
for p in policies:
    means, cis = [], []
    for load in sorted(nom.load.unique()):
        vals = nom[(nom.policy == p) & (nom.load == load)].on_time_fraction_pct
        m, h = mean_ci(vals); means.append(m); cis.append(h)
    loads = sorted(nom.load.unique())
    ax.errorbar(loads, means, yerr=cis, color=colors[p], marker=markers[p],
                linewidth=1.8, capsize=2.5, label=labels[p])
ax.set(xlabel=r"Normalized uplink load $\rho$", ylabel="Mean on-time completion fraction (%)",
       ylim=(0, 102))
ax.grid(True, linestyle=":", alpha=.55)
ax.legend(ncol=2, fontsize=8.5, frameon=True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGURES / f"fig_load_sweep.{ext}", dpi=400, bbox_inches="tight")
plt.close(fig)

# Fig. 2: stress-point means and 95% CIs.
stress = df[(df.load == 1.2) & (df.tau_ms == 50.0) & (df.cv == 1.0)]
means, cis = zip(*(mean_ci(stress[stress.policy == p].on_time_fraction_pct) for p in policies))
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(policies))
bars = ax.bar(x, means, yerr=cis, capsize=4, color=[colors[p] for p in policies],
              edgecolor="black", linewidth=.7)
ax.set_xticks(x, ["No margin", "Fixed", "Rolling\nquantile", "Stabilized"], fontsize=9)
ax.set_ylabel("Mean on-time completion fraction (%)")
ax.set_ylim(58, 72)
ax.grid(axis="y", linestyle=":", alpha=.55)
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+.35, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGURES / f"fig_stress_comparison.{ext}", dpi=400, bbox_inches="tight")
plt.close(fig)

# Fig. 3: stabilized-minus-no-margin heatmaps; cell values are percentage points.
stab = df[df.policy == "DA-BALS-Stabilized"]
nomarg = df[df.policy == "DA-BALS-NoMargin"]
paired = stab.merge(nomarg, on=["seed", "load", "tau_ms", "cv"], suffixes=("_s", "_n"))
paired["delta"] = paired.on_time_fraction_pct_s - paired.on_time_fraction_pct_n
agg = paired.groupby(["load", "tau_ms", "cv"]).delta.mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), sharey=True)
v = max(abs(agg.delta.min()), abs(agg.delta.max()))
for ax, cv in zip(axes, sorted(agg.cv.unique())):
    part = agg[agg.cv == cv].pivot(index="tau_ms", columns="load", values="delta")
    im = ax.imshow(part.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_title(f"CV = {cv:g}")
    ax.set_xticks(range(len(part.columns)), [f"{q:g}" for q in part.columns])
    ax.set_yticks(range(len(part.index)), [f"{t:g}" for t in part.index])
    ax.set_xlabel(r"Load $\rho$")
    for i in range(part.shape[0]):
        for j in range(part.shape[1]):
            val = part.iloc[i,j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > .55*v else "black")
axes[0].set_ylabel(r"Telemetry delay $\tau$ (ms)")
cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.03)
cb.set_label("Stabilized - no margin (percentage points)")
fig.subplots_adjust(left=.07, right=.91, bottom=.18, top=.86, wspace=.18)
for ext in ("pdf", "png"):
    fig.savefig(FIGURES / f"fig_nomargin_heatmap.{ext}", dpi=400, bbox_inches="tight")
plt.close(fig)

print("Generated three manuscript figures from robustness_run_metrics.csv")
