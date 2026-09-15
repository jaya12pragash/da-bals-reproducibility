import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / "results" / "journal_raw_experiments.csv")

# Reservoir for grouped seed metrics
processed_metrics = []

# Group data by Seed, Load Point, and Admission Policy to compute per-run metrics
grouped = df.groupby(['seed', 'load', 'policy'])

for (seed, load, policy), group in grouped:
    total_requests = len(group)
    successful_requests = group[group['status'] == 'Success']
    num_success = len(successful_requests)
    
    # 1. Compute On-Time Completion Fraction (%)
    goodput = (num_success / total_requests) * 100.0 if total_requests > 0 else 0.0
    
    # 2. Compute 95th Percentile Latency for Successful Requests
    if num_success > 0:
        p95_latency = np.percentile(successful_requests['latency_ms'], 95)
    else:
        p95_latency = np.nan  # No successful requests to sample delay
        
    processed_metrics.append({
        "seed": seed,
        "load": load,
        "policy": policy,
        "goodput": goodput,
        "p95_latency": p95_latency
    })

# Convert to DataFrame for statistical summary
df_summary = pd.DataFrame(processed_metrics)

# Structure final data matrices for plotting
load_points = sorted(df_summary['load'].unique())
policies = ["FIFO", "STS", "DOA", "DA-BALS", "Oracle"]
colors = {"FIFO": "#d62728", "STS": "#ff7f0e", "DOA": "#bcbd22", "DA-BALS": "#1f77b4", "Oracle": "#2ca02c"}
markers = {"FIFO": "o", "STS": "^", "DOA": "v", "DA-BALS": "s", "Oracle": "D"}

# Initialize multi-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

for policy in policies:
    policy_df = df_summary[df_summary['policy'] == policy]
    
    means_goodput = []
    cis_goodput = []
    means_latency = []
    cis_latency = []
    
    for load in load_points:
        load_df = policy_df[policy_df['load'] == load]
        
        # --- Statistical Calculations for Goodput ---
        gp_data = load_df['goodput'].dropna()
        if len(gp_data) > 1:
            means_goodput.append(gp_data.mean())
            # 95% Confidence Interval using t-distribution
            sem = stats.sem(gp_data)
            ci = sem * stats.t.ppf((1 + 0.95) / 2., len(gp_data) - 1)
            cis_goodput.append(ci)
        else:
            means_goodput.append(np.nan)
            cis_goodput.append(0)
            
        # --- Statistical Calculations for p95 Latency ---
        lat_data = load_df['p95_latency'].dropna()
        if len(lat_data) > 1:
            means_latency.append(lat_data.mean())
            sem = stats.sem(lat_data)
            ci = sem * stats.t.ppf((1 + 0.95) / 2., len(lat_data) - 1)
            cis_latency.append(ci)
        else:
            means_latency.append(np.nan)
            cis_latency.append(0)

    # --- Plot Figure (a): On-Time Goodput with Shaded 95% CI ---
    means_goodput = np.array(means_goodput)
    cis_goodput = np.array(cis_goodput)
    ax1.plot(load_points, means_goodput, label=policy, color=colors[policy], marker=markers[policy], linewidth=2, markersize=5)
    ax1.fill_between(load_points, means_goodput - cis_goodput, means_goodput + cis_goodput, color=colors[policy], alpha=0.15)

    # --- Plot Figure (b): Tail Latency with Shaded 95% CI ---
    means_latency = np.array(means_latency)
    cis_latency = np.array(cis_latency)
    ax2.plot(load_points, means_latency, label=policy, color=colors[policy], marker=markers[policy], linewidth=2, markersize=5)
    ax2.fill_between(load_points, means_latency - cis_latency, means_latency + cis_latency, color=colors[policy], alpha=0.15)

# Style panel (a)
ax1.set_xlabel(r'Normalized Uplink Traffic Load Factor ($\rho$)', fontsize=11)
ax1.set_ylabel('Mean On-Time Completion Fraction (%)', fontsize=11)
ax1.set_title('(a) On-Time Completion Fraction', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlim(0.15, 1.25)
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, shadow=False)

# Style panel (b)
ax2.set_xlabel(r'Normalized Uplink Traffic Load Factor ($\rho$)', fontsize=11)
ax2.set_ylabel('Mean 95th Percentile Latency (ms)', fontsize=11)
ax2.set_title('(b) Conditional Successful-Request Latency', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlim(0.15, 1.25)
ax2.set_ylim(0, 60)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
# Save directly into your workspace repository
plt.savefig(PROJECT_ROOT / 'figures' / 'fig_2.png', dpi=300, bbox_inches='tight')
print("Statistical summary processing complete. Main scaling plot exported as 'fig_2.png'.")
