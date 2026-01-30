import os
import numpy as np
import pandas as pd
from scipy import stats

# =========================
# Configuration
# =========================
RF_PATH  = r"rf_dynamic_outputs_v2\oos_torpor_predictions_and_residuals.csv"
GAM_PATH = r"gam_outputs\oos_torpor_predictions_and_residuals.csv"


OUT_DIR = "statistical_validation_results"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.05
VO2_OBS_TOL = 1e-9


# =========================
# Helper Functions
# =========================
def safe_rmse(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))

def safe_mae_from_errors(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.mean(np.abs(errors)))

def wilcoxon_two_sided_paired(a, b):
    """
    Paired Wilcoxon signed-rank test (two-sided).
    H0: median(a - b) = 0
    Returns: (statistic, p_value, median(a-b))
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b

    if np.allclose(diff, 0):
        return np.nan, 1.0, 0.0

    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p), float(np.median(diff))
    except Exception as e:
        print(f"⚠️ Wilcoxon test failed: {e}")
        return np.nan, np.nan, np.nan

def describe_winner(median_diff_rf_minus_gam, p_value, alpha=0.05):
    """
    median_diff_rf_minus_gam: median( MAE_RF - MAE_GAM )
      negative => RF lower error (RF better)
      positive => GAM lower error (GAM better)
    """
    if np.isnan(p_value):
        return "Unable to determine (test failed)."
    if p_value >= alpha:
        return "No statistically detectable difference in MAE between models."
    if median_diff_rf_minus_gam < 0:
        return "Random Forest has significantly lower MAE than GAM."
    if median_diff_rf_minus_gam > 0:
        return "GAM has significantly lower MAE than Random Forest."
    return "Statistically significant but median difference is ~0 (check data)."


# =========================
# Flexible column standardiser
# =========================
def standardise_cols(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Make sure we end up with: mouse_id, time, VO2_obs, VO2_pred
    Handles common variants from your RF script + other pipelines.
    """
    rename_map = {}

    # time
    if "time" not in df.columns:
        if "Time" in df.columns: rename_map["Time"] = "time"
        elif "t" in df.columns: rename_map["t"] = "time"

    # mouse_id
    if "mouse_id" not in df.columns:
        if "Subject_ID" in df.columns: rename_map["Subject_ID"] = "mouse_id"
        elif "Mouse_ID" in df.columns: rename_map["Mouse_ID"] = "mouse_id"
        elif "id" in df.columns: rename_map["id"] = "mouse_id"

    # VO2_obs
    if "VO2_obs" not in df.columns:
        if "VO2" in df.columns: rename_map["VO2"] = "VO2_obs"
        elif "vo2" in df.columns: rename_map["vo2"] = "VO2_obs"
        elif "y_true" in df.columns: rename_map["y_true"] = "VO2_obs"

    # VO2_pred
    if "VO2_pred" not in df.columns:
        if "pred" in df.columns: rename_map["pred"] = "VO2_pred"
        elif "y_pred" in df.columns: rename_map["y_pred"] = "VO2_pred"
        elif "VO2_hat" in df.columns: rename_map["VO2_hat"] = "VO2_pred"

    df = df.rename(columns=rename_map)

    required = ["mouse_id", "time", "VO2_obs", "VO2_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name} dataset missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[required].dropna().copy()
    out["mouse_id"] = out["mouse_id"].astype(str)
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["VO2_obs"] = pd.to_numeric(out["VO2_obs"], errors="coerce")
    out["VO2_pred"] = pd.to_numeric(out["VO2_pred"], errors="coerce")
    out = out.dropna().copy()

    return out


# =========================
# 1) Load Data
# =========================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

for p, label in [(RF_PATH, "RF"), (GAM_PATH, "GAM")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ {label} file not found: {p}")

rf_raw  = pd.read_csv(RF_PATH)
gam_raw = pd.read_csv(GAM_PATH)

print(f"✅ Loaded RF:  {len(rf_raw)} rows")
print(f"✅ Loaded GAM: {len(gam_raw)} rows")


# =========================
# 2) Standardise + Clean
# =========================
print("\n" + "=" * 60)
print("STANDARDISING + CLEANING")
print("=" * 60)

rf  = standardise_cols(rf_raw, "RF")
gam = standardise_cols(gam_raw, "GAM")

print(f"RF cleaned:  {len(rf)} rows, mice={rf['mouse_id'].nunique()}")
print(f"GAM cleaned: {len(gam)} rows, mice={gam['mouse_id'].nunique()}")


# =========================
# 3) Merge on same timepoints
# =========================
print("\n" + "=" * 60)
print("MERGING PREDICTIONS (mouse_id + time)")
print("=" * 60)

rf_m  = rf.rename(columns={"VO2_obs": "VO2_obs_rf", "VO2_pred": "VO2_pred_rf"})
gam_m = gam.rename(columns={"VO2_obs": "VO2_obs_gam", "VO2_pred": "VO2_pred_gam"})

merged = pd.merge(rf_m, gam_m, on=["mouse_id", "time"], how="inner")

if len(merged) == 0:
    raise ValueError("❌ Merge empty. Check mouse_id/time alignment between RF and GAM outputs.")

vo2_obs_max_abs_diff = float(np.max(np.abs(merged["VO2_obs_rf"] - merged["VO2_obs_gam"])))
print(f"✅ Merged: {len(merged)} rows across {merged['mouse_id'].nunique()} mice")
print(f"Max |VO2_obs_rf - VO2_obs_gam| = {vo2_obs_max_abs_diff:.6g}")

if vo2_obs_max_abs_diff > VO2_OBS_TOL:
    print("⚠️ WARNING: VO2_obs differs after merge (rounding/processing mismatch). Using RF obs as reference.")

merged["VO2_obs"] = merged["VO2_obs_rf"]


# =========================
# 4) Errors
# =========================
merged["error_rf"]  = merged["VO2_pred_rf"]  - merged["VO2_obs"]
merged["error_gam"] = merged["VO2_pred_gam"] - merged["VO2_obs"]

merged["abs_error_rf"]  = merged["error_rf"].abs()
merged["abs_error_gam"] = merged["error_gam"].abs()


# =========================
# 5) MAIN: Per-mouse comparison
# =========================
print("\n" + "=" * 60)
print("STATISTICAL TEST: PER-MOUSE COMPARISON (RECOMMENDED)")
print("=" * 60)

per_mouse = merged.groupby("mouse_id").agg(
    n_timepoints=("time", "size"),
    mae_rf=("abs_error_rf", "mean"),
    mae_gam=("abs_error_gam", "mean"),
    rmse_rf=("error_rf", lambda x: safe_rmse(x)),
    rmse_gam=("error_gam", lambda x: safe_rmse(x))
).reset_index()

per_mouse["mae_diff_rf_minus_gam"]  = per_mouse["mae_rf"]  - per_mouse["mae_gam"]
per_mouse["rmse_diff_rf_minus_gam"] = per_mouse["rmse_rf"] - per_mouse["rmse_gam"]

n_mice = len(per_mouse)

# Wilcoxon on per-mouse MAE: compare RF vs GAM (paired)
w_stat, w_pval, median_diff_rf_minus_gam_wilcox = wilcoxon_two_sided_paired(
    per_mouse["mae_rf"].values,
    per_mouse["mae_gam"].values
)

median_diff_rf_minus_gam = float(np.median(per_mouse["mae_diff_rf_minus_gam"]))
winner_sentence = describe_winner(median_diff_rf_minus_gam, w_pval, alpha=ALPHA)

print(per_mouse.to_string(index=False))
print("\n" + "=" * 60)
print("MAIN RESULT: RF vs GAM (per-mouse MAE)")
print("=" * 60)
print(f"N mice: {n_mice}")
print(f"RF  MAE mean={per_mouse['mae_rf'].mean():.4f}, median={per_mouse['mae_rf'].median():.4f}")
print(f"GAM MAE mean={per_mouse['mae_gam'].mean():.4f}, median={per_mouse['mae_gam'].median():.4f}")
print(f"Median Δ(MAE) = median(RF - GAM) = {median_diff_rf_minus_gam:.4f}")
print(f"Wilcoxon (two-sided): W={w_stat}, p={w_pval:.4f}")
print(f"Conclusion: {winner_sentence}")
print("=" * 60)


# =========================
# 6) Point-level stats (reference only)
# =========================
print("\n" + "=" * 60)
print("POINT-LEVEL STATISTICS (REFERENCE ONLY)")
print("=" * 60)
print("⚠️ Pooled timepoints violate independence; report per-mouse as main result.\n")

mae_rf_all  = safe_mae_from_errors(merged["error_rf"].values)
mae_gam_all = safe_mae_from_errors(merged["error_gam"].values)
rmse_rf_all = safe_rmse(merged["error_rf"].values)
rmse_gam_all = safe_rmse(merged["error_gam"].values)

print(f"Overall pooled MAE  RF : {mae_rf_all:.4f}")
print(f"Overall pooled MAE  GAM: {mae_gam_all:.4f}")
print(f"Overall pooled RMSE RF : {rmse_rf_all:.4f}")
print(f"Overall pooled RMSE GAM: {rmse_gam_all:.4f}")


# =========================
# 7) Save outputs
# =========================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

results_table = pd.DataFrame([{
    "test": "RF vs GAM (per-mouse MAE, Wilcoxon two-sided)",
    "n_mice": n_mice,
    "mean_mae_rf": float(per_mouse["mae_rf"].mean()),
    "mean_mae_gam": float(per_mouse["mae_gam"].mean()),
    "median_mae_rf": float(per_mouse["mae_rf"].median()),
    "median_mae_gam": float(per_mouse["mae_gam"].median()),
    "median_diff_rf_minus_gam": median_diff_rf_minus_gam,
    "wilcoxon_statistic": w_stat,
    "p_value": w_pval,
    "significant_p05": bool(w_pval < ALPHA),
    "vo2_obs_max_abs_diff_after_merge": vo2_obs_max_abs_diff
}])

results_table.to_csv(os.path.join(OUT_DIR, "main_statistical_test_results.csv"), index=False)
per_mouse.to_csv(os.path.join(OUT_DIR, "per_mouse_errors.csv"), index=False)
merged.to_csv(os.path.join(OUT_DIR, "merged_predictions_all_timepoints.csv"), index=False)

print(f"✅ Saved: {OUT_DIR}/main_statistical_test_results.csv")
print(f"✅ Saved: {OUT_DIR}/per_mouse_errors.csv")
print(f"✅ Saved: {OUT_DIR}/merged_predictions_all_timepoints.csv")


# =========================
# 8) Write-up paragraph
# =========================
sig_phrase = "was significantly different from" if (w_pval < ALPHA) else "did not differ significantly from"

report_text = f"""
METHODS:
Statistical comparison between Random Forest and GAM was performed using a paired
Wilcoxon signed-rank test (two-sided) on per-mouse prediction error, treating each mouse
as the unit of replication (N={n_mice}). Mean Absolute Error (MAE) was computed per mouse
on matched held-out torpor-like observations. Statistical significance was assessed at α={ALPHA:.2f}.

RESULTS:
Per-mouse MAE for GAM (mean={per_mouse['mae_gam'].mean():.4f}, median={per_mouse['mae_gam'].median():.4f})
{sig_phrase} Random Forest (mean={per_mouse['mae_rf'].mean():.4f}, median={per_mouse['mae_rf'].median():.4f};
Wilcoxon W={w_stat}, p={w_pval:.4f}). The median paired MAE difference (RF − GAM) was
{median_diff_rf_minus_gam:.4f} (negative valokay lets ues indicate lower RF error; positive values indicate lower GAM error).
Conclusion: {winner_sentence}
"""

with open(os.path.join(OUT_DIR, "report_summary.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)


print(f"✅ Saved: {OUT_DIR}/report_summary.txt")
print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE")
print("=" * 60)
