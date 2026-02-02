
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer


XLSX_PATH = "Organized data.xlsx"

SHEETS = {
    "tcore": ("Tcore torpor-like", "Tcore nontorpor"),
    "vo2":   ("VO2 torpor-like", "VO2 nontorpor"),
    "act":   ("activity torpor-like", "ativity nontorpor"),  # typo matches file
    "hr":    ("Heart Rate torpor-like", "Heart Rate nontorpor"),
}

OUTDIR = "rf_dynamic_outputs_v2"
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_STATE = 0

# Feature switches
INCLUDE_TIME = True
USE_LAG_FEATURES = True
USE_DELTA_FEATURES = False
CLIP_TORPOR_TO_TRAIN_RANGE = False

TIME_MIN = None
TIME_MAX = None


TUNE_N_ITER = 20
TUNE_INNER_SPLITS = 5


RF_FALLBACK_PARAMS = dict(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=3,
    max_features=0.7,
)


CB = {
    "blue":   "tab:blue",
    "orange": "tab:orange",
    "green":  "tab:green",
    "red":    "tab:red",
    "purple": "tab:purple",
    "grey":   "tab:gray",
    "black":  "black",
}

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})



def read_timeseries_sheet(xlsx_path: str, sheet_name: str, value_name: str) -> pd.DataFrame:
    """
    Sheet format:
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


def build_master(xlsx_path: str) -> pd.DataFrame:
    parts = []
    for condition_label, idx in [("torpor_like", 0), ("nontorpor", 1)]:
        vo2 = read_timeseries_sheet(xlsx_path, SHEETS["vo2"][idx], "VO2")
        vo2["Condition"] = condition_label

        tcore = read_timeseries_sheet(xlsx_path, SHEETS["tcore"][idx], "Tcore")
        act = read_timeseries_sheet(xlsx_path, SHEETS["act"][idx], "Activity")
        hr = read_timeseries_sheet(xlsx_path, SHEETS["hr"][idx], "HeartRate")

        key = ["Time", "Subject_ID"]
        df = vo2.merge(tcore[key + ["Tcore"]], on=key, how="left")
        df = df.merge(act[key + ["Activity"]], on=key, how="left")
        df = df.merge(hr[key + ["HeartRate"]], on=key, how="left")

        parts.append(df)

    master = pd.concat(parts, ignore_index=True)

    master["Time"] = pd.to_numeric(master["Time"], errors="coerce")
    for c in ["VO2", "Tcore", "Activity", "HeartRate"]:
        master[c] = pd.to_numeric(master[c], errors="coerce")

    master["Sex"] = master["Sex"].fillna("U").astype(str)
    master["Condition"] = master["Condition"].astype(str)
    master["Subject_ID"] = master["Subject_ID"].astype(str)

    if TIME_MIN is not None:
        master = master[master["Time"] >= float(TIME_MIN)]
    if TIME_MAX is not None:
        master = master[master["Time"] <= float(TIME_MAX)]

    return master.reset_index(drop=True)



def add_predictor_lags(df: pd.DataFrame, group_cols=("Condition", "Subject_ID")) -> pd.DataFrame:
    """
    Adds lag1 (and optionally deltas) for predictors within each mouse within each condition.
    Does NOT create VO2 lag.
    """
    df = df.sort_values(list(group_cols) + ["Time"]).copy()

    for col in ["Tcore", "HeartRate", "Activity"]:
        df[f"{col}_lag1"] = df.groupby(list(group_cols))[col].shift(1)
        if USE_DELTA_FEATURES:
            df[f"d{col}"] = df[col] - df[f"{col}_lag1"]

    return df


def make_feature_cols() -> list:
    base = []
    if INCLUDE_TIME:
        base.append("Time")
    base += ["Tcore", "HeartRate", "Activity", "Sex_Encoded"]

    if USE_LAG_FEATURES:
        base += ["Tcore_lag1", "HeartRate_lag1", "Activity_lag1"]
        if USE_DELTA_FEATURES:
            base += ["dTcore", "dHeartRate", "dActivity"]

    return base


def clip_to_train_range(df_in: pd.DataFrame, train_ref: pd.DataFrame, cols: list) -> pd.DataFrame:
    df_out = df_in.copy()
    for c in cols:
        lo = train_ref[c].min()
        hi = train_ref[c].max()
        df_out[c] = df_out[c].clip(lo, hi)
    return df_out



def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def neg_rmse_scorer():
    return make_scorer(lambda yt, yp: -rmse(yt, yp), greater_is_better=True)


def tune_rf_on_non_torpor(train_non: pd.DataFrame, feature_cols: list) -> dict:
    """
    Tune RF using GroupKFold across mice on the current fold's NON-TORPOR training data.
    This does NOT use torpor data at all.
    Returns best params dict.
    """
    X = train_non[feature_cols]
    y = train_non["VO2"].values
    groups = train_non["Subject_ID"].values

    base = RandomForestRegressor(
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


    param_dist = {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [None, 6, 10, 14, 18],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "max_features": ["sqrt", 0.5, 0.7, 1.0],
    }

    n_groups = len(np.unique(groups))
    n_splits = min(TUNE_INNER_SPLITS, n_groups) if n_groups >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=TUNE_N_ITER,
        scoring=neg_rmse_scorer(),
        cv=gkf.split(X, y, groups=groups),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    search.fit(X, y)
    best = dict(search.best_params_)

    print("    [tuning] Best params:", best)
    print("    [tuning] Best CV score (neg-RMSE):", float(search.best_score_))
    return best



def run_loso_rf(df_all: pd.DataFrame) -> None:
    df = df_all.copy()

    df["Sex_Encoded"] = df["Sex"].map({"M": 0, "F": 1}).fillna(0).astype(int)


    if USE_LAG_FEATURES:
        df = add_predictor_lags(df)

    feature_cols = make_feature_cols()


    non = df[df["Condition"] == "nontorpor"].copy()
    non = non.dropna(subset=["VO2"] + feature_cols).copy()

    print(f"Non-torpor rows: {len(non)}  Non-torpor mice: {non['Subject_ID'].nunique()}")
    print("Features:", feature_cols)

    X_non = non[feature_cols].copy()
    y_non = non["VO2"].values
    groups = non["Subject_ID"].values

    logo = LeaveOneGroupOut()

    oos_non_rows = []
    oos_torpor_rows = []
    fold_rmses = []

    torpor_metrics_rows = []
    best_params_rows = []

    for fold_i, (tr, te) in enumerate(logo.split(X_non, y_non, groups=groups), start=1):
        test_mouse = pd.Series(groups[te]).iloc[0]

        train_non = non.iloc[tr].copy()
        test_non = non.iloc[te].copy()

        print(f"\nfold {fold_i:02d} | held-out mouse: {test_mouse}")


        try:
            best_params = tune_rf_on_non_torpor(train_non, feature_cols)
        except Exception as e:
            print("    [tuning] FAILED, using fallback params. Error:", repr(e))
            best_params = dict(RF_FALLBACK_PARAMS)

        best_params_rows.append({"fold": fold_i, "held_out_mouse": str(test_mouse), **best_params})


        rf = RandomForestRegressor(
            **best_params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        counts = train_non["Subject_ID"].value_counts()
        wtr = train_non["Subject_ID"].map(lambda g: 1.0 / counts[g]).values.astype(float)

        rf.fit(train_non[feature_cols], train_non["VO2"].values, sample_weight=wtr)

        pred_non = rf.predict(test_non[feature_cols])
        fold_rmse = rmse(test_non["VO2"].values, pred_non)
        fold_rmses.append(fold_rmse)
        print(f"    OOS non-torpor RMSE={fold_rmse:.4f}")

        tmp_non = test_non[["Time", "Subject_ID"]].copy()
        tmp_non["Group"] = "Non-torpor"
        tmp_non["VO2_obs"] = test_non["VO2"].values
        tmp_non["VO2_pred"] = pred_non
        oos_non_rows.append(tmp_non)


        tor = df[(df["Condition"] == "torpor_like") & (df["Subject_ID"] == test_mouse)].copy()
        if len(tor) == 0:
            continue

        tor = tor.dropna(subset=["VO2"] + feature_cols).copy()
        if len(tor) == 0:
            continue

        tor_for_pred = tor.copy()
        if CLIP_TORPOR_TO_TRAIN_RANGE:
            clip_cols = [c for c in feature_cols if c not in ["Sex_Encoded"]]
            tor_for_pred = clip_to_train_range(tor_for_pred, train_non, clip_cols)

        pred_tor = rf.predict(tor_for_pred[feature_cols])

        tmp_tor = tor[["Time", "Subject_ID"]].copy()
        tmp_tor["Group"] = "Torpor-like"
        tmp_tor["VO2_obs"] = tor["VO2"].values
        tmp_tor["VO2_pred"] = pred_tor
        tmp_tor["Residual"] = tmp_tor["VO2_obs"] - tmp_tor["VO2_pred"]
        oos_torpor_rows.append(tmp_tor)

        tor_rmse = rmse(tmp_tor["VO2_obs"].values, tmp_tor["VO2_pred"].values)
        tor_mae = mae(tmp_tor["VO2_obs"].values, tmp_tor["VO2_pred"].values)

        torpor_metrics_rows.append({
            "fold": int(fold_i),
            "Subject_ID": str(test_mouse),
            "torpor_n": int(len(tmp_tor)),
            "torpor_rmse": float(tor_rmse),
            "torpor_mae": float(tor_mae),
        })
        print(f"    OOS torpor RMSE={tor_rmse:.4f}  MAE={tor_mae:.4f}")

    oos_non = pd.concat(oos_non_rows, ignore_index=True) if oos_non_rows else pd.DataFrame()
    oos_tor = pd.concat(oos_torpor_rows, ignore_index=True) if oos_torpor_rows else pd.DataFrame()
    torpor_metrics = pd.DataFrame(torpor_metrics_rows)
    best_params_df = pd.DataFrame(best_params_rows)

    non_csv = os.path.join(OUTDIR, "oos_non_torpor_predictions.csv")
    tor_csv = os.path.join(OUTDIR, "oos_torpor_predictions_and_residuals.csv")
    metrics_csv = os.path.join(OUTDIR, "torpor_oos_metrics_by_mouse.csv")
    params_csv = os.path.join(OUTDIR, "best_params_by_fold.csv")
    summary_txt = os.path.join(OUTDIR, "metrics_summary.txt")

    oos_non.to_csv(non_csv, index=False)
    oos_tor.to_csv(tor_csv, index=False)
    torpor_metrics.to_csv(metrics_csv, index=False)
    best_params_df.to_csv(params_csv, index=False)

    overall_tor_rmse = np.nan
    overall_tor_mae = np.nan
    if len(oos_tor) > 0:
        overall_tor_rmse = rmse(oos_tor["VO2_obs"].values, oos_tor["VO2_pred"].values)
        overall_tor_mae = mae(oos_tor["VO2_obs"].values, oos_tor["VO2_pred"].values)

    with open(summary_txt, "w") as f:
        f.write("=== Non-torpor LOSO (held-out non-torpor mouse) ===\n")
        f.write(f"Non-torpor LOSO RMSE mean={np.mean(fold_rmses):.6f}, std={np.std(fold_rmses):.6f}\n\n")

        f.write("=== Torpor-like OOS (held-out mouse, torpor segment) ===\n")
        if len(torpor_metrics) > 0:
            f.write(f"Per-mouse torpor RMSE mean={torpor_metrics['torpor_rmse'].mean():.6f}, "
                    f"std={torpor_metrics['torpor_rmse'].std(ddof=0):.6f}\n")
            f.write(f"Per-mouse torpor MAE  mean={torpor_metrics['torpor_mae'].mean():.6f}, "
                    f"std={torpor_metrics['torpor_mae'].std(ddof=0):.6f}\n")
        if len(oos_tor) > 0:
            f.write(f"\nPooled OOS torpor RMSE={overall_tor_rmse:.6f}\n")
            f.write(f"Pooled OOS torpor MAE ={overall_tor_mae:.6f}\n")
        else:
            f.write("No OOS torpor rows available.\n")

        f.write("\n=== Hyperparameters ===\n")
        f.write(f"Saved per-fold tuned params to: {params_csv}\n")

    print("\n======================")
    print(f"Non-torpor LOSO RMSE mean={np.mean(fold_rmses):.4f}, std={np.std(fold_rmses):.4f}")
    if len(oos_tor) > 0:
        print(f"OOS Torpor pooled: RMSE={overall_tor_rmse:.4f}, MAE={overall_tor_mae:.4f}")
    print("======================\n")

    print(f"Saved: {non_csv}")
    print(f"Saved: {tor_csv}")
    print(f"Saved: {metrics_csv}")
    print(f"Saved: {params_csv}")
    print(f"Saved: {summary_txt}")


    if len(oos_non) > 0:
        plt.figure(figsize=(7, 7))
        plt.scatter(
            oos_non["VO2_obs"].values,
            oos_non["VO2_pred"].values,
            alpha=0.55,
            s=18,
            color=CB["blue"],
            edgecolor=CB["black"],
            linewidth=0.25,
        )
        mn = float(min(oos_non["VO2_obs"].min(), oos_non["VO2_pred"].min()))
        mx = float(max(oos_non["VO2_obs"].max(), oos_non["VO2_pred"].max()))
        plt.plot([mn, mx], [mn, mx], linestyle="--", color=CB["grey"], linewidth=1.5, label="y=x")
        plt.xlabel("VO₂ actual (non-torpor, OOS)")
        plt.ylabel("VO₂ predicted (non-torpor, OOS)")
        plt.title("RF (tuned within LOSO): OOS predicted vs actual (non-torpor)")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(OUTDIR, "figB_non_torpor_pred_vs_actual.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")

    if len(oos_non) > 0 and len(oos_tor) > 0:
        g_non = (oos_non.groupby("Time", as_index=False)
                      .agg(VO2_obs_mean=("VO2_obs", "mean"),
                           VO2_pred_mean=("VO2_pred", "mean")))
        g_tor = (oos_tor.groupby("Time", as_index=False)
                      .agg(VO2_obs_mean=("VO2_obs", "mean"),
                           VO2_pred_mean=("VO2_pred", "mean")))

        plt.figure(figsize=(12, 5))

        plt.plot(
            g_non["Time"], g_non["VO2_obs_mean"],
            label="Non-torpor Actual",
            color=CB["blue"], linewidth=2
        )
        plt.plot(
            g_non["Time"], g_non["VO2_pred_mean"],
            label="Non-torpor Predicted",
            color=CB["blue"], linestyle="--", linewidth=2
        )

        plt.plot(
            g_tor["Time"], g_tor["VO2_obs_mean"],
            label="Torpor Actual",
            color=CB["orange"], linewidth=2
        )
        plt.plot(
            g_tor["Time"], g_tor["VO2_pred_mean"],
            label="Torpor Predicted (by baseline)",
            color=CB["orange"], linestyle="--", linewidth=2
        )

        plt.xlabel("Time (min)")
        plt.ylabel("Mean VO₂")
        plt.title("Random Forest (tuned within LOSO): mean VO₂ over time (OOS only)")
        plt.legend(ncol=2)
        plt.tight_layout()
        out = os.path.join(OUTDIR, "figA_mean_vo2_over_time_actual_vs_pred.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")


    if len(oos_tor) > 0:
        gt = (oos_tor.groupby("Time", as_index=False)
                    .agg(mean_res=("Residual", "mean"),
                         std_res=("Residual", "std"),
                         n=("Residual", "size")))
        gt["sem"] = gt["std_res"] / np.sqrt(gt["n"].clip(lower=1))

        plt.figure(figsize=(12, 4))
        plt.plot(
            gt["Time"].values,
            gt["mean_res"].values,
            color=CB["purple"],
            linewidth=2,
            label="Mean residual"
        )
        plt.fill_between(
            gt["Time"].values,
            (gt["mean_res"] - gt["sem"]).values,
            (gt["mean_res"] + gt["sem"]).values,
            alpha=0.2,
            color=CB["purple"],
        )
        plt.axhline(0, linestyle="--", color=CB["grey"], linewidth=1.5)
        plt.xlabel("Time (min)")
        plt.ylabel("Residual VO₂ (obs - expected)")
        plt.title("Torpor-like: mean residual over time (OOS only) ± SEM")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(OUTDIR, "figC_torpor_mean_residual_over_time_sem.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    df_master = build_master(XLSX_PATH)
    run_loso_rf(df_master)
