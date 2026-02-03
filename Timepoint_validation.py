# DipValidation_TimepointAccuracy.py
import os
import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# =========================
# CONFIG
# =========================
RF_CSV   = r"rf_dynamic_outputs_v2\oos_torpor_predictions_and_residuals.csv"
GAM_CSV  = r"gam_outputs\oos_torpor_predictions_and_residuals.csv"
GAMM_CSV = r"gamm_outputs\oos_torpor_predictions_and_residuals.csv"

OUTDIR = "statistical_validation_results"
os.makedirs(OUTDIR, exist_ok=True)

TIME_WINDOW = (35, 45)
ALPHA = 0.05


# =========================
# HELPERS
# =========================
def _standardize_cols(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    d = df.copy()

    # mouse id
    if "mouse_id" in d.columns:
        d["mouse_id"] = d["mouse_id"].astype(str)
    elif "Subject_ID" in d.columns:
        d = d.rename(columns={"Subject_ID": "mouse_id"})
        d["mouse_id"] = d["mouse_id"].astype(str)
    else:
        raise ValueError(f"[{model_name}] missing mouse_id / Subject_ID")

    # time
    if "time" in d.columns:
        pass
    elif "Time" in d.columns:
        d = d.rename(columns={"Time": "time"})
    else:
        raise ValueError(f"[{model_name}] missing time / Time")

    # observed
    if "VO2_obs" in d.columns:
        pass
    elif "VO2" in d.columns:
        d = d.rename(columns={"VO2": "VO2_obs"})
    else:
        raise ValueError(f"[{model_name}] missing VO2_obs / VO2")

    # predicted
    pred_candidates = [
        "VO2_pred", "vo2_pred", "y_pred", "pred",
        "VO2_pred_rf", "VO2_pred_gam", "VO2_pred_gamm"
    ]
    pred_col = None
    for c in pred_candidates:
        if c in d.columns:
            pred_col = c
            break
    if pred_col is None:
        for c in d.columns:
            if "pred" in c.lower():
                pred_col = c
                break
    if pred_col is None:
        raise ValueError(f"[{model_name}] could not find prediction column")

    if pred_col != "VO2_pred":
        d = d.rename(columns={pred_col: "VO2_pred"})

    for c in ["time", "VO2_obs", "VO2_pred"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["mouse_id", "time", "VO2_obs", "VO2_pred"]).copy()
    return d[["mouse_id", "time", "VO2_obs", "VO2_pred"]]


def _window(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return df[(df["time"] >= t0) & (df["time"] <= t1)].copy()


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x * x)))


def _mae(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.mean(np.abs(x)))


def _paired_stats_over_time(per_time_wide: pd.DataFrame, metric_name: str):
    """
    per_time_wide must have columns: time, rf, gam, gamm
    uses timepoints as paired blocks
    """
    wide = per_time_wide[["time", "rf", "gam", "gamm"]].dropna().copy()
    nT = len(wide)

    chi2, p_f = friedmanchisquare(wide["rf"].values, wide["gam"].values, wide["gamm"].values)

    pairs = [("rf", "gam"), ("rf", "gamm"), ("gam", "gamm")]
    rows = []
    pvals = []
    for a, b in pairs:
        stat, p = wilcoxon(wide[a].values, wide[b].values, alternative="two-sided", zero_method="wilcox")
        pvals.append(p)
        rows.append(dict(
            metric=metric_name,
            comparison=f"{a} vs {b}",
            wilcoxon_stat=float(stat),
            wilcoxon_p_raw=float(p),
            median_diff_a_minus_b=float(np.median(wide[a].values - wide[b].values)),
            mean_diff_a_minus_b=float(np.mean(wide[a].values - wide[b].values)),
            n_timepoints=int(nT),
        ))

    rej, p_holm, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
    for i in range(len(rows)):
        rows[i]["wilcoxon_p_holm"] = float(p_holm[i])
        rows[i]["reject_holm_alpha"] = bool(rej[i])

    summary = dict(
        metric=metric_name,
        friedman_chi2=float(chi2),
        friedman_p=float(p_f),
        n_timepoints=int(nT),
        medians={
            "rf": float(np.median(wide["rf"].values)),
            "gam": float(np.median(wide["gam"].values)),
            "gamm": float(np.median(wide["gamm"].values)),
        }
    )
    return summary, pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():
    print("============================================================")
    print("DIP WINDOW TIMEPOINT VALIDATION (unit = time)")
    print(f"TIME_WINDOW = {TIME_WINDOW}")
    print("============================================================")

    rf   = _standardize_cols(pd.read_csv(RF_CSV),   "RF")
    gam  = _standardize_cols(pd.read_csv(GAM_CSV),  "GAM")
    gamm = _standardize_cols(pd.read_csv(GAMM_CSV), "GAMM")

    t0, t1 = TIME_WINDOW
    rf_w   = _window(rf, t0, t1)
    gam_w  = _window(gam, t0, t1)
    gamm_w = _window(gamm, t0, t1)

    common_mice = sorted(set(rf_w["mouse_id"]) & set(gam_w["mouse_id"]) & set(gamm_w["mouse_id"]))
    rf_w   = rf_w[rf_w["mouse_id"].isin(common_mice)].copy()
    gam_w  = gam_w[gam_w["mouse_id"].isin(common_mice)].copy()
    gamm_w = gamm_w[gamm_w["mouse_id"].isin(common_mice)].copy()

    panel = rf_w.rename(columns={"VO2_pred": "pred_rf"}) \
        .merge(gam_w.rename(columns={"VO2_pred": "pred_gam"}), on=["mouse_id", "time", "VO2_obs"], how="inner") \
        .merge(gamm_w.rename(columns={"VO2_pred": "pred_gamm"}), on=["mouse_id", "time", "VO2_obs"], how="inner")

    common_times = sorted(panel["time"].unique().tolist())

    print(f"Common mice: {len(common_mice)} -> {common_mice}")
    print(f"Common timepoints in window: {len(common_times)} -> {common_times}")

    panel["e_rf"]   = panel["pred_rf"]   - panel["VO2_obs"]
    panel["e_gam"]  = panel["pred_gam"]  - panel["VO2_obs"]
    panel["e_gamm"] = panel["pred_gamm"] - panel["VO2_obs"]

    rows = []
    for t in common_times:
        sub = panel[panel["time"] == t].copy()
        rows.append({
            "time": float(t),
            "n_mice_at_t": int(len(sub)),
            "mae_rf": _mae(sub["e_rf"].values),
            "mae_gam": _mae(sub["e_gam"].values),
            "mae_gamm": _mae(sub["e_gamm"].values),
            "rmse_rf": _rmse(sub["e_rf"].values),
            "rmse_gam": _rmse(sub["e_gam"].values),
            "rmse_gamm": _rmse(sub["e_gamm"].values),
            "bias_rf": float(np.mean(sub["e_rf"].values)),
            "bias_gam": float(np.mean(sub["e_gam"].values)),
            "bias_gamm": float(np.mean(sub["e_gamm"].values)),
        })

    per_time = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    per_time_path = os.path.join(OUTDIR, "dip_window_timepoint_metrics_3models.csv")
    per_time.to_csv(per_time_path, index=False)
    print(f"\nSaved per-time metrics: {per_time_path}")

    # Wide for tests
    per_time_mae = per_time[["time"]].copy()
    per_time_mae["rf"] = per_time["mae_rf"]
    per_time_mae["gam"] = per_time["mae_gam"]
    per_time_mae["gamm"] = per_time["mae_gamm"]

    per_time_rmse = per_time[["time"]].copy()
    per_time_rmse["rf"] = per_time["rmse_rf"]
    per_time_rmse["gam"] = per_time["rmse_gam"]
    per_time_rmse["gamm"] = per_time["rmse_gamm"]

    per_time_bias = per_time[["time"]].copy()
    per_time_bias["rf"] = per_time["bias_rf"]
    per_time_bias["gam"] = per_time["bias_gam"]
    per_time_bias["gamm"] = per_time["bias_gamm"]

    print("\n====================")
    print("TIMEPOINT-LEVEL HYPOTHESIS TESTS (unit = time)")
    print("====================")

    summaries = []
    posthocs = []

    for name, wide in [
        ("mae_over_time", per_time_mae),
        ("rmse_over_time", per_time_rmse),
        ("bias_over_time", per_time_bias),
    ]:
        summary, posthoc = _paired_stats_over_time(wide, name)
        summaries.append(summary)
        posthocs.append(posthoc)
        med = summary["medians"]
        print(f"{name:>14s} | rf={med['rf']:.4f}  gam={med['gam']:.4f}  gamm={med['gamm']:.4f} "
              f"| Friedman p={summary['friedman_p']:.3g} (nT={summary['n_timepoints']})")

    posthoc_df = pd.concat(posthocs, ignore_index=True)
    posthoc_path = os.path.join(OUTDIR, "dip_window_timepoint_friedman_posthoc_3models.csv")
    posthoc_df.to_csv(posthoc_path, index=False)
    print(f"\nSaved stats: {posthoc_path}")

    summary_df = pd.DataFrame([{
        "metric": s["metric"],
        "friedman_chi2": s["friedman_chi2"],
        "friedman_p": s["friedman_p"],
        "n_timepoints": s["n_timepoints"],
        "median_rf": s["medians"]["rf"],
        "median_gam": s["medians"]["gam"],
        "median_gamm": s["medians"]["gamm"],
    } for s in summaries])
    summary_path = os.path.join(OUTDIR, "dip_window_timepoint_summary_3models.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
