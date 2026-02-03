""""
complete_residual_diagnostics_LOSO_fair.py

Fair residual diagnostics across:
1) Linear baseline: VO2 ~ Tcore   (trained on non-torpor, LOSO-by-mouse, tested on torpor)
2) Linear baseline: VO2 ~ Tcore + HeartRate (same LOSO)
3) Random Forest: uses your existing LOSO torpor CSV
4) GAMM: uses your existing torpor CSV

Outputs:
- Per-model residual diagnostics (3-panel)
- Residuals vs time plot (torpor only) per model
- Summary CSV of RMSE/MAE/bias
- Optional combined comparison figure

IMPORTANT:
- Update the PATHS below to match your files.
- This script assumes the Excel layout matches your RF reader (row0 mouse IDs, row1 sex, row2+ time+values),
  consistent with your rf script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# SETTINGS
# =========================
SHOW_PLOTS = True  # set False if you only want saved PNGs

LINEAR_DATA_XLSX = r"C:\Users\alici\Downloads\Organized data.xlsx"
RF_TORPOR_CSV = r"C:\Users\alici\PycharmProjects\Intensive-Care\rf_dynamic_outputs_v2\oos_torpor_predictions_and_residuals.csv"
GAMM_TORPOR_CSV = r"C:\Users\alici\PycharmProjects\Intensive-Care\gamm_oos_torpor_predictions.csv"

OUTDIR = "combined_residual_diagnostics"
os.makedirs(OUTDIR, exist_ok=True)

# Excel sheets (match your project naming)
SHEETS = {
    ("non", "Tcore"): "Tcore nontorpor",
    ("torpor", "Tcore"): "Tcore torpor-like",
    ("non", "VO2"): "VO2 nontorpor",
    ("torpor", "VO2"): "VO2 torpor-like",
    ("non", "HR"): "Heart Rate nontorpor",
    ("torpor", "HR"): "Heart Rate torpor-like",
}

# If you want to restrict time window for residual plots (e.g. dip window), set these:
TIME_MIN = None
TIME_MAX = None


# ============================================================
# Helpers: metrics + plotting
# ============================================================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def apply_time_filter(df: pd.DataFrame, time_col="Time") -> pd.DataFrame:
    out = df.copy()
    if TIME_MIN is not None:
        out = out[out[time_col] >= float(TIME_MIN)]
    if TIME_MAX is not None:
        out = out[out[time_col] <= float(TIME_MAX)]
    return out

def create_residual_plot(yobs, ypred, model_name, rmse_val, mae_val, output_path, show=False):
    resid = yobs - ypred

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # (1) Residuals vs Predicted
    axes[0].scatter(ypred, resid, alpha=0.6, s=18)
    axes[0].axhline(0, linestyle="--", color="black")
    axes[0].set_xlabel("Predicted VO₂ (ŷ)")
    axes[0].set_ylabel("Residual (y - ŷ)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(alpha=0.3)

    # (2) Residuals vs Observed
    axes[1].scatter(yobs, resid, alpha=0.6, s=18)
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set_xlabel("Observed VO₂ (y)")
    axes[1].set_ylabel("Residual (y - ŷ)")
    axes[1].set_title("Residuals vs Observed")
    axes[1].grid(alpha=0.3)

    # (3) Histogram of residuals
    axes[2].hist(resid, bins=30, alpha=0.85)
    axes[2].axvline(0, linestyle="--", color="black")
    axes[2].set_xlabel("Residual (y - ŷ)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual distribution")
    axes[2].grid(alpha=0.3)

    plt.suptitle(
        f"{model_name} - Residual diagnostics\nRMSE={rmse_val:.3f}  MAE={mae_val:.3f}",
        fontsize=14, fontweight="bold", y=1.05
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

    print(f"✓ Saved: {output_path}")
    print(f"  RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
    print(f"  Mean residual (bias): {resid.mean():.4f}, Std residual: {resid.std():.4f}\n")


def residuals_vs_time_plot(df_pred: pd.DataFrame, model_name: str, output_path: str, show=False):
    """
    df_pred must include columns: Time, Residual
    Plots mean residual vs time with SEM band.
    """
    dfp = apply_time_filter(df_pred, time_col="Time")

    g = (dfp.groupby("Time", as_index=False)
            .agg(mean_res=("Residual", "mean"),
                 std_res=("Residual", "std"),
                 n=("Residual", "size")))
    g["sem"] = g["std_res"] / np.sqrt(g["n"].clip(lower=1))

    plt.figure(figsize=(12, 4))
    plt.plot(g["Time"].values, g["mean_res"].values)
    plt.fill_between(
        g["Time"].values,
        (g["mean_res"] - g["sem"]).values,
        (g["mean_res"] + g["sem"]).values,
        alpha=0.2
    )
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Time (min)")
    plt.ylabel("Mean residual (obs - pred)")
    plt.title(f"{model_name}: Mean residual over time (± SEM)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
    print(f"✓ Saved: {output_path}\n")


# ============================================================
# Excel reader consistent with your RF script
# ============================================================
def read_timeseries_sheet(xlsx_path: str, sheet_name: str, value_name: str) -> pd.DataFrame:
    """
    Expected sheet format (same as your RF script):
      row 0: mouse IDs in columns 1..
      row 1: sex labels (M/F) in columns 1..
      row 2+: time in col 0 and values in columns 1..
    Returns long-form: Time, Subject_ID, Sex, <value_name>
    """
    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    mouse_ids = df_raw.iloc[0, 1:].astype(str).tolist()
    sex_labels = df_raw.iloc[1, 1:].astype(str).tolist()
    sex_map = dict(zip(mouse_ids, sex_labels))

    time = pd.to_numeric(df_raw.iloc[2:, 0], errors="coerce")
    vals = df_raw.iloc[2:, 1:]
    vals.columns = mouse_ids
    vals = vals.apply(pd.to_numeric, errors="coerce")

    df_long = vals.copy()
    df_long.insert(0, "Time", time.values)
    df_long = df_long.melt(id_vars=["Time"], var_name="Subject_ID", value_name=value_name)
    df_long["Sex"] = df_long["Subject_ID"].map(sex_map)

    return df_long


def build_master_linear(xlsx_path: str) -> pd.DataFrame:
    """
    Builds a master table with both conditions for linear baselines:
    Columns: Condition, Time, Subject_ID, VO2, Tcore, HeartRate, Sex
    """
    parts = []
    for cond_label in ["non", "torpor"]:
        vo2 = read_timeseries_sheet(xlsx_path, SHEETS[(cond_label, "VO2")], "VO2")
        tcore = read_timeseries_sheet(xlsx_path, SHEETS[(cond_label, "Tcore")], "Tcore")
        hr = read_timeseries_sheet(xlsx_path, SHEETS[(cond_label, "HR")], "HeartRate")

        key = ["Time", "Subject_ID"]
        df = vo2.merge(tcore[key + ["Tcore"]], on=key, how="left")
        df = df.merge(hr[key + ["HeartRate"]], on=key, how="left")
        df["Condition"] = cond_label
        parts.append(df)

    master = pd.concat(parts, ignore_index=True)

    # numeric
    master["Time"] = pd.to_numeric(master["Time"], errors="coerce")
    for c in ["VO2", "Tcore", "HeartRate"]:
        master[c] = pd.to_numeric(master[c], errors="coerce")

    master["Subject_ID"] = master["Subject_ID"].astype(str)

    # time filter optional
    master = apply_time_filter(master, time_col="Time")

    return master.reset_index(drop=True)


# ============================================================
# LOSO baselines (trained on NON, tested on TORPOR of held-out mouse)
# ============================================================
def run_loso_linear_baseline(df_master: pd.DataFrame, feature_cols: list, model_tag: str) -> pd.DataFrame:
    """
    For each held-out mouse:
      - fit LinearRegression on NON data from other mice
      - predict TORPOR rows for held-out mouse
    Returns pooled OOS torpor predictions with residuals.
    """
    assert "Condition" in df_master.columns
    assert "Subject_ID" in df_master.columns

    non = df_master[df_master["Condition"] == "non"].copy()
    tor = df_master[df_master["Condition"] == "torpor"].copy()

    # only keep rows with full data
    non = non.dropna(subset=["VO2"] + feature_cols).copy()
    tor = tor.dropna(subset=["VO2"] + feature_cols).copy()

    mice = sorted(non["Subject_ID"].unique().tolist())
    out_rows = []

    print(f"\nLOSO Linear baseline: {model_tag}")
    print(f"  Non rows: {len(non)} | Torpor rows: {len(tor)} | Mice: {len(mice)}")

    for i, test_mouse in enumerate(mice, start=1):
        train_non = non[non["Subject_ID"] != test_mouse].copy()
        test_tor = tor[tor["Subject_ID"] == test_mouse].copy()

        if len(test_tor) == 0 or len(train_non) == 0:
            continue

        Xtr = train_non[feature_cols].values
        ytr = train_non["VO2"].values

        Xte = test_tor[feature_cols].values
        yte = test_tor["VO2"].values

        model = LinearRegression()
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        tmp = test_tor[["Time", "Subject_ID"]].copy()
        tmp["Model"] = model_tag
        tmp["VO2_obs"] = yte
        tmp["VO2_pred"] = yhat
        tmp["Residual"] = tmp["VO2_obs"] - tmp["VO2_pred"]
        out_rows.append(tmp)

        fold_rmse = rmse(yte, yhat)
        fold_mae = mae(yte, yhat)
        print(f"  fold {i:02d} | held-out mouse={test_mouse} | torpor_n={len(tmp)} | RMSE={fold_rmse:.4f} | MAE={fold_mae:.4f}")

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    return out


# ============================================================
# Load RF / GAMM OOS torpor CSVs in a standard format
# ============================================================
def load_model_csv(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Try to standardize columns
    # RF file: Time, Subject_ID, VO2_obs, VO2_pred, Residual (already)
    # GAMM file: you said it has VO2_obs, VO2_pred, and (hopefully) Time, mouse_id/Subject_ID
    colmap = {}
    if "mouse_id" in df.columns and "Subject_ID" not in df.columns:
        colmap["mouse_id"] = "Subject_ID"
    if "time" in df.columns and "Time" not in df.columns:
        colmap["time"] = "Time"
    if "VO2_pred_gamm" in df.columns and "VO2_pred" not in df.columns:
        colmap["VO2_pred_gamm"] = "VO2_pred"
    if "VO2_pred_rf" in df.columns and "VO2_pred" not in df.columns:
        colmap["VO2_pred_rf"] = "VO2_pred"

    if colmap:
        df = df.rename(columns=colmap)

    required = {"VO2_obs", "VO2_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_name} CSV missing required columns: {missing}. Columns found: {list(df.columns)}")

    # Residual
    if "Residual" not in df.columns:
        df["Residual"] = df["VO2_obs"] - df["VO2_pred"]

    # Subject/time optional but strongly preferred for time plot
    if "Subject_ID" not in df.columns:
        df["Subject_ID"] = "unknown"
    if "Time" not in df.columns:
        df["Time"] = np.nan

    df["Model"] = model_name
    df["Subject_ID"] = df["Subject_ID"].astype(str)
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    df = apply_time_filter(df, time_col="Time")
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    # 1) Build master from Excel for linear baselines
    print("=" * 60)
    print("Building master dataset for linear baselines (from Excel)...")
    print("=" * 60)
    df_master = build_master_linear(LINEAR_DATA_XLSX)

    # 2) Linear baselines (fair LOSO on non -> predict torpor)
    lin1 = run_loso_linear_baseline(
        df_master=df_master,
        feature_cols=["Tcore"],
        model_tag="Linear (VO2 ~ Tcore) [LOSO non→torpor]"
    )
    lin2 = run_loso_linear_baseline(
        df_master=df_master,
        feature_cols=["Tcore", "HeartRate"],
        model_tag="Linear (VO2 ~ Tcore + HR) [LOSO non→torpor]"
    )

    # Save their OOS torpor predictions (so everything is reproducible)
    lin1_csv = os.path.join(OUTDIR, "linear_temp_only_oos_torpor_predictions.csv")
    lin2_csv = os.path.join(OUTDIR, "linear_temp_hr_oos_torpor_predictions.csv")
    if len(lin1) > 0:
        lin1.to_csv(lin1_csv, index=False)
        print(f"\n✓ Saved: {lin1_csv}")
    if len(lin2) > 0:
        lin2.to_csv(lin2_csv, index=False)
        print(f"✓ Saved: {lin2_csv}\n")

    # 3) Load RF & GAMM torpor OOS (existing)
    models = []

    if len(lin1) > 0:
        models.append(lin1)
    if len(lin2) > 0:
        models.append(lin2)

    rf_exists = os.path.exists(RF_TORPOR_CSV)
    gamm_exists = os.path.exists(GAMM_TORPOR_CSV)

    if rf_exists:
        df_rf = load_model_csv(RF_TORPOR_CSV, "Random Forest [LOSO torpor OOS]")
        models.append(df_rf)
    else:
        print(f"⚠️ RF CSV not found: {RF_TORPOR_CSV}")

    if gamm_exists:
        df_gamm = load_model_csv(GAMM_TORPOR_CSV, "GAMM [torpor OOS]")
        models.append(df_gamm)
    else:
        print(f"⚠️ GAMM CSV not found: {GAMM_TORPOR_CSV}")

    if not models:
        print("❌ No models available to plot. Check paths / CSV generation.")
        return

    # 4) Per-model residual diagnostics + residual-vs-time
    summary_rows = []

    print("=" * 60)
    print("Creating residual diagnostics...")
    print("=" * 60)

    for dfm in models:
        name = str(dfm["Model"].iloc[0])

        yobs = dfm["VO2_obs"].values.astype(float)
        ypred = dfm["VO2_pred"].values.astype(float)

        r = rmse(yobs, ypred)
        a = mae(yobs, ypred)
        bias = float(np.mean(yobs - ypred))

        # 3-panel residual diagnostics
        out_png = os.path.join(OUTDIR, f"{safe_filename(name)}_residuals.png")
        create_residual_plot(
            yobs=yobs,
            ypred=ypred,
            model_name=name,
            rmse_val=r,
            mae_val=a,
            output_path=out_png,
            show=SHOW_PLOTS
        )

        # residual vs time (if Time exists)
        if dfm["Time"].notna().any():
            out_time_png = os.path.join(OUTDIR, f"{safe_filename(name)}_residuals_vs_time.png")
            residuals_vs_time_plot(
                df_pred=dfm[["Time", "Residual"]].copy(),
                model_name=name,
                output_path=out_time_png,
                show=SHOW_PLOTS
            )

        summary_rows.append({
            "Model": name,
            "Dataset": "Torpor-like OOS (fair LOSO for linear; existing OOS for RF/GAMM)",
            "RMSE": r,
            "MAE": a,
            "MeanResidual_Bias(obs-pred)": bias,
            "N": int(len(dfm))
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("MAE")
    print("\n" + "=" * 60)
    print("SUMMARY (lower is better for MAE/RMSE)")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(OUTDIR, "model_comparison_summary_fair_LOSO.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved: {summary_csv}")

    # 5) Optional combined comparison figure (same layout you wanted)
    print("\n" + "=" * 60)
    print("Creating combined comparison figure...")
    print("=" * 60)

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(16, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, dfm in enumerate(models):
        name = str(dfm["Model"].iloc[0])
        yobs = dfm["VO2_obs"].values.astype(float)
        ypred = dfm["VO2_pred"].values.astype(float)
        resid = yobs - ypred

        axes[row, 0].scatter(ypred, resid, alpha=0.6, s=18)
        axes[row, 0].axhline(0, linestyle="--", color="black")
        axes[row, 0].set_ylabel(f"{name}\nResidual", fontsize=9, fontweight="bold")
        axes[row, 0].set_xlabel("Predicted VO₂")
        axes[row, 0].grid(alpha=0.3)

        axes[row, 1].scatter(yobs, resid, alpha=0.6, s=18)
        axes[row, 1].axhline(0, linestyle="--", color="black")
        axes[row, 1].set_xlabel("Observed VO₂")
        axes[row, 1].grid(alpha=0.3)

        axes[row, 2].hist(resid, bins=30, alpha=0.85)
        axes[row, 2].axvline(0, linestyle="--", color="black")
        axes[row, 2].set_xlabel("Residual")
        axes[row, 2].grid(alpha=0.3)

    axes[0, 0].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Residuals vs Observed", fontsize=12, fontweight="bold")
    axes[0, 2].set_title("Residual Distribution", fontsize=12, fontweight="bold")

    plt.suptitle("Model Comparison: Residual Diagnostics (Fair LOSO Torpor OOS)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    combined_path = os.path.join(OUTDIR, "all_models_residual_comparison_fair_LOSO.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()
    print(f"✓ Saved: {combined_path}")

    print("\n" + "=" * 60)
    print("All residual diagnostics complete!")
    print(f"Output directory: {OUTDIR}")
    print("=" * 60)


def safe_filename(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", ".", " "):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip().replace(" ", "_")[:180]


if __name__ == "__main__":
    main()
