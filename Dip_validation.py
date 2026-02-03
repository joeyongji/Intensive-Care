# DipValidation_DipMorphology.py
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

    if "mouse_id" in d.columns:
        d["mouse_id"] = d["mouse_id"].astype(str)
    elif "Subject_ID" in d.columns:
        d = d.rename(columns={"Subject_ID": "mouse_id"})
        d["mouse_id"] = d["mouse_id"].astype(str)
    else:
        raise ValueError(f"[{model_name}] missing mouse_id / Subject_ID")

    if "time" in d.columns:
        pass
    elif "Time" in d.columns:
        d = d.rename(columns={"Time": "time"})
    else:
        raise ValueError(f"[{model_name}] missing time / Time")

    if "VO2_obs" in d.columns:
        pass
    elif "VO2" in d.columns:
        d = d.rename(columns={"VO2": "VO2_obs"})
    else:
        raise ValueError(f"[{model_name}] missing VO2_obs / VO2")

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


def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def _dip_metrics_for_mouse(df_mouse: pd.DataFrame) -> dict:
    d = df_mouse.sort_values("time").copy()

    y = d["VO2_obs"].to_numpy(float)
    p = d["VO2_pred"].to_numpy(float)
    t = d["time"].to_numpy(float)

    # amplitude (depth range)
    amp_obs = float(np.max(y) - np.min(y))
    amp_pred = float(np.max(p) - np.min(p))
    amp_err = abs(amp_pred - amp_obs)

    # nadir time
    tmin_obs = float(t[np.argmin(y)])
    tmin_pred = float(t[np.argmin(p)])
    tmin_err = abs(tmin_pred - tmin_obs)

    # nadir magnitude error at observed nadir time
    idx_obs_min = int(np.argmin(y))
    nadir_mag_err = abs(p[idx_obs_min] - y[idx_obs_min])

    # dip AUC relative to baseline (start-of-window observed)
    baseline = float(y[0])
    auc_obs = float(np.trapezoid(y - baseline, t))
    auc_pred = float(np.trapezoid(p - baseline, t))
    auc_err = abs(auc_pred - auc_obs)

    # shape similarity: demeaned correlation and DTW
    y0 = y - np.mean(y)
    p0 = p - np.mean(p)
    denom = (np.linalg.norm(y0) * np.linalg.norm(p0))
    corr = float(np.dot(y0, p0) / denom) if denom > 0 else np.nan
    dtw = _dtw_distance(y0, p0)

    # dip detection: does predicted drop exceed threshold (adaptive)
    thr = 0.2 * amp_obs
    drop_pred = float(p[0] - np.min(p))
    dip_success = int(drop_pred >= thr)

    return dict(
        amp_err=amp_err,
        tmin_err=tmin_err,
        nadir_mag_err=nadir_mag_err,
        auc_err=auc_err,
        corr=corr,
        dtw=dtw,
        dip_success=dip_success,
        amp_obs=amp_obs,
        amp_pred=amp_pred,
        tmin_obs=tmin_obs,
        tmin_pred=tmin_pred,
    )


def _paired_stats(df_metrics: pd.DataFrame, metric: str):
    wide = df_metrics.pivot(index="mouse_id", columns="model", values=metric).dropna()
    for c in ["rf", "gam", "gamm"]:
        if c not in wide.columns:
            raise ValueError(f"Missing model column '{c}' for metric '{metric}'")

    chi2, p_f = friedmanchisquare(wide["rf"].values, wide["gam"].values, wide["gamm"].values)

    pairs = [("rf", "gam"), ("rf", "gamm"), ("gam", "gamm")]
    rows = []
    pvals = []
    for a, b in pairs:
        stat, p = wilcoxon(wide[a].values, wide[b].values, alternative="two-sided", zero_method="wilcox")
        pvals.append(p)
        rows.append(dict(
            metric=metric,
            comparison=f"{a} vs {b}",
            wilcoxon_stat=float(stat),
            wilcoxon_p_raw=float(p),
            median_diff_a_minus_b=float(np.median(wide[a].values - wide[b].values)),
            mean_diff_a_minus_b=float(np.mean(wide[a].values - wide[b].values)),
            n=int(len(wide)),
        ))

    rej, p_holm, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
    for i in range(len(rows)):
        rows[i]["wilcoxon_p_holm"] = float(p_holm[i])
        rows[i]["reject_holm_alpha"] = bool(rej[i])

    summary = dict(
        metric=metric,
        friedman_chi2=float(chi2),
        friedman_p=float(p_f),
        n_mice=int(len(wide)),
        medians={k: float(np.median(wide[k].values)) for k in ["rf", "gam", "gamm"]},
    )
    return summary, pd.DataFrame(rows)


def _cochrans_q_binary(wide_binary: pd.DataFrame):
    """
    Cochran's Q for k=3 related samples (binary outcomes).
    wide_binary: index=mouse, columns=[rf,gam,gamm] with 0/1
    """
    X = wide_binary[["rf", "gam", "gamm"]].values.astype(int)
    n, k = X.shape
    col_sums = X.sum(axis=0)
    row_sums = X.sum(axis=1)

    T = col_sums.sum()
    num = (k - 1) * (k * np.sum(col_sums**2) - T**2)
    den = k * T - np.sum(row_sums**2)
    Q = float(num / den) if den != 0 else np.nan

    # p-value via chi-square with df=k-1
    from scipy.stats import chi2
    p = float(1.0 - chi2.cdf(Q, df=k - 1)) if np.isfinite(Q) else np.nan
    return Q, p


# =========================
# MAIN
# =========================
def main():
    print("============================================================")
    print("DIP-FOCUSED VALIDATION (dip morphology per mouse)")
    print(f"TIME_WINDOW = {TIME_WINDOW}")
    print("============================================================")

    rf   = _standardize_cols(pd.read_csv(RF_CSV),   "RF")
    gam  = _standardize_cols(pd.read_csv(GAM_CSV),  "GAM")
    gamm = _standardize_cols(pd.read_csv(GAMM_CSV), "GAMM")

    t0, t1 = TIME_WINDOW
    rf_w   = _window(rf, t0, t1)
    gam_w  = _window(gam, t0, t1)
    gamm_w = _window(gamm, t0, t1)

    # strict common mice
    mice = sorted(set(rf_w["mouse_id"]) & set(gam_w["mouse_id"]) & set(gamm_w["mouse_id"]))
    rf_w   = rf_w[rf_w["mouse_id"].isin(mice)].copy()
    gam_w  = gam_w[gam_w["mouse_id"].isin(mice)].copy()
    gamm_w = gamm_w[gamm_w["mouse_id"].isin(mice)].copy()
    print(f"Common mice: {len(mice)} -> {mice}")

    # also enforce common timepoints per mouse across models by inner merge on (mouse_id,time)
    panel = rf_w.rename(columns={"VO2_pred": "pred_rf"}) \
        .merge(gam_w.rename(columns={"VO2_pred": "pred_gam"}), on=["mouse_id", "time", "VO2_obs"], how="inner") \
        .merge(gamm_w.rename(columns={"VO2_pred": "pred_gamm"}), on=["mouse_id", "time", "VO2_obs"], how="inner")

    # rebuild model-specific dfs with identical (mouse_id,time,VO2_obs) support
    rf_w2 = panel[["mouse_id", "time", "VO2_obs"]].copy()
    rf_w2["VO2_pred"] = panel["pred_rf"].values
    gam_w2 = panel[["mouse_id", "time", "VO2_obs"]].copy()
    gam_w2["VO2_pred"] = panel["pred_gam"].values
    gamm_w2 = panel[["mouse_id", "time", "VO2_obs"]].copy()
    gamm_w2["VO2_pred"] = panel["pred_gamm"].values

    # per-mouse metrics per model
    all_rows = []
    for model_name, dfm in [("rf", rf_w2), ("gam", gam_w2), ("gamm", gamm_w2)]:
        for mid, g in dfm.groupby("mouse_id"):
            m = _dip_metrics_for_mouse(g)
            m["mouse_id"] = mid
            m["model"] = model_name
            all_rows.append(m)

    metrics = pd.DataFrame(all_rows)
    metrics_path = os.path.join(OUTDIR, "dip_window_morphology_metrics_per_mouse_3models.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved per-mouse metrics: {metrics_path}")

    # Metrics list
    metric_list = ["dip_success", "corr", "dtw", "amp_err", "tmin_err", "nadir_mag_err", "auc_err"]

    print("\n====================")
    print("DIP MORPHOLOGY TESTS (unit = mouse)")
    print("====================")

    summaries = []
    posthoc_rows = []

    # Binary outcome: Cochran's Q
    wide_bin = metrics.pivot(index="mouse_id", columns="model", values="dip_success").dropna()
    Q, pQ = _cochrans_q_binary(wide_bin)
    print(f"{'dip_success':>14s} | rf={np.median(wide_bin['rf']):.4f}  gam={np.median(wide_bin['gam']):.4f}  "
          f"gamm={np.median(wide_bin['gamm']):.4f} | cochrans_q p={pQ:.3g} (n={len(wide_bin)})")

    # Save Cochran Q result
    cochran_path = os.path.join(OUTDIR, "dip_window_cochransQ_dip_success.csv")
    pd.DataFrame([{
        "metric": "dip_success",
        "cochrans_q": Q,
        "p_value": pQ,
        "n_mice": int(len(wide_bin)),
        "success_rate_rf": float(wide_bin["rf"].mean()),
        "success_rate_gam": float(wide_bin["gam"].mean()),
        "success_rate_gamm": float(wide_bin["gamm"].mean()),
    }]).to_csv(cochran_path, index=False)

    # Continuous metrics: Friedman + Wilcoxon
    for metric in [m for m in metric_list if m != "dip_success"]:
        summary, posthoc = _paired_stats(metrics[["mouse_id", "model", metric]], metric)
        summaries.append(summary)
        posthoc_rows.append(posthoc)
        med = summary["medians"]
        print(f"{metric:>14s} | rf={med['rf']:.4f}  gam={med['gam']:.4f}  gamm={med['gamm']:.4f} "
              f"| friedman p={summary['friedman_p']:.3g} (n={summary['n_mice']})")

    posthoc_df = pd.concat(posthoc_rows, ignore_index=True) if posthoc_rows else pd.DataFrame()
    posthoc_path = os.path.join(OUTDIR, "dip_window_morphology_friedman_posthoc_3models.csv")
    posthoc_df.to_csv(posthoc_path, index=False)
    print(f"\nSaved stats: {posthoc_path}")
    print(f"Saved Cochran Q: {cochran_path}")

    summary_df = pd.DataFrame([{
        "metric": s["metric"],
        "friedman_chi2": s["friedman_chi2"],
        "friedman_p": s["friedman_p"],
        "n_mice": s["n_mice"],
        "median_rf": s["medians"]["rf"],
        "median_gam": s["medians"]["gam"],
        "median_gamm": s["medians"]["gamm"],
    } for s in summaries])
    summary_path = os.path.join(OUTDIR, "dip_window_morphology_summary_3models.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
