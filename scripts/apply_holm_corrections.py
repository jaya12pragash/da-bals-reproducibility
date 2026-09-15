"""Holm corrections for robustness-only and secondary pooled comparisons."""
from pathlib import Path
import pandas as pd
from scipy import stats
from run_holdout_evaluation import holm_adjust

ROOT = Path(__file__).resolve().parents[1]

def compare(frame, label):
    rows=[]
    for (load,tau,cv), block in frame.groupby(["load","tau_ms","cv"]):
        a=block[block.policy=="DA-BALS-Stabilized"][["seed","on_time_fraction_pct"]]
        b=block[block.policy=="DA-BALS-NoMargin"][["seed","on_time_fraction_pct"]]
        p=a.merge(b,on="seed",suffixes=("_stabilized","_nomargin"),validate="one_to_one")
        delta=p.on_time_fraction_pct_stabilized-p.on_time_fraction_pct_nomargin
        rows.append({"analysis":label,"load":load,"tau_ms":tau,"cv":cv,"pairs":len(delta),
                     "mean_delta_pp":delta.mean(),"p_value":stats.ttest_1samp(delta,0).pvalue})
    result=pd.DataFrame(rows)
    result["holm_p_value"]=holm_adjust(result.p_value)
    result["holm_significant"]=result.holm_p_value.lt(.05)
    result["direction"] = result.mean_delta_pp.map(lambda x: "positive" if x>0 else "negative")
    return result


def main():
    holdout=pd.read_csv(ROOT / "results" / "holdout_run_metrics.csv")
    robust=pd.read_csv(ROOT / "results" / "robustness_run_metrics.csv")
    robust_result=compare(robust,"robustness_only")
    pooled_result=compare(pd.concat([holdout,robust],ignore_index=True),"secondary_pooled")
    result=pd.concat([robust_result,pooled_result],ignore_index=True)
    output=ROOT / "results"; result.to_csv(output/"holm_nomargin_results.csv",index=False)
    with (output/"holm_report.txt").open("w") as report:
        for name,group in result.groupby("analysis",sort=False):
            positive=((group.holm_significant)&(group.direction=="positive")).sum()
            negative=((group.holm_significant)&(group.direction=="negative")).sum()
            report.write(f"{name}: Holm-significant positive={positive}, negative={negative}, total={len(group)}\n")
    print((output/"holm_report.txt").read_text())


if __name__=="__main__":
    main()
