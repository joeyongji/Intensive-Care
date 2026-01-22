import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. 配置与参数
# ==========================================
XLSX_PATH = "Organized data1.xlsx"  # 确保文件名正确

SHEETS = {
    "tcore": ("Tcore torpor-like", "Tcore nontorpor"),
    "vo2": ("VO2 torpor-like", "VO2 nontorpor"),
    "act": ("activity torpor-like", "ativity nontorpor"),
    "hr": ("Heart Rate torpor-like", "Heart Rate nontorpor"),  # 这里实际是 RQ
}

OUTDIR = "rf_ablation_outputs"
os.makedirs(OUTDIR, exist_ok=True)

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=3,
    max_features=0.7,
    random_state=0,
    n_jobs=-1,
)

USE_LAG_FEATURES = True
CLIP_TORPOR_TO_TRAIN_RANGE = True


# ==========================================
# 2. 数据读取与处理 Helper Functions
# ==========================================

def read_timeseries_sheet(xlsx_path, sheet_name, value_name):
    if not os.path.exists(xlsx_path): return pd.DataFrame()
    try:
        df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    except ValueError:
        return pd.DataFrame()

    if df_raw.empty: return pd.DataFrame()

    # 你的逻辑：Row 0=ID, Row 1=Sex, Row 2+=Data
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


def build_master(xlsx_path):
    parts = []
    if not os.path.exists(xlsx_path): return pd.DataFrame()

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names

    for condition_label, idx in [("torpor_like", 0), ("nontorpor", 1)]:
        s_vo2 = SHEETS["vo2"][idx]
        s_tcore = SHEETS["tcore"][idx]
        s_act = SHEETS["act"][idx]
        s_hr = SHEETS["hr"][idx]

        if s_vo2 not in sheet_names: continue

        vo2 = read_timeseries_sheet(xlsx_path, s_vo2, "VO2")
        vo2["Condition"] = condition_label

        tcore = read_timeseries_sheet(xlsx_path, s_tcore, "Tcore")
        act = read_timeseries_sheet(xlsx_path, s_act, "Activity")
        hr = read_timeseries_sheet(xlsx_path, s_hr, "HeartRate")

        if vo2.empty: continue

        key = ["Time", "Subject_ID"]
        df = vo2
        if not tcore.empty: df = df.merge(tcore[key + ["Tcore"]], on=key, how="left")
        if not act.empty: df = df.merge(act[key + ["Activity"]], on=key, how="left")
        if not hr.empty: df = df.merge(hr[key + ["HeartRate"]], on=key, how="left")

        parts.append(df)

    if not parts: return pd.DataFrame()

    master = pd.concat(parts, ignore_index=True)
    master["Time"] = pd.to_numeric(master["Time"], errors="coerce")
    for c in ["VO2", "Tcore", "Activity", "HeartRate"]:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce")

    if "Sex" in master.columns:
        master["Sex"] = master["Sex"].fillna("U")
    master["Condition"] = master["Condition"].astype(str)
    master["Subject_ID"] = master["Subject_ID"].astype(str)

    return master.reset_index(drop=True)


def add_predictor_lags(df):
    if df.empty: return df
    df = df.sort_values(["Condition", "Subject_ID", "Time"]).copy()
    for col in ["Tcore", "HeartRate", "Activity"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby(["Condition", "Subject_ID"])[col].shift(1)
            df[f"d{col}"] = df[col] - df[f"{col}_lag1"]
    return df


def clip_to_train_range(df_in, train_ref, cols):
    df_out = df_in.copy()
    for c in cols:
        if c in df_in.columns and c in train_ref.columns:
            lo = train_ref[c].min()
            hi = train_ref[c].max()
            df_out[c] = df_out[c].clip(lo, hi)
    return df_out


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ==========================================
# 3. 核心部分：Ablation Runner
# ==========================================

def run_ablation_rf(df_master):
    if df_master.empty:
        print("No data available.")
        return

    df = df_master.copy()
    # 预处理：性别编码
    if "Sex" in df.columns:
        df["Sex_Encoded"] = df["Sex"].map({"M": 0, "F": 1}).fillna(0).astype(int)
    else:
        df["Sex_Encoded"] = 0

    if USE_LAG_FEATURES:
        df = add_predictor_lags(df)

    # === 定义消融模型 ===
    # 注意：HeartRate 其实是 RQ
    models = [
        ("Model 1: Time + Tcore", ["Time", "Tcore", "Sex_Encoded"]),
        ("Model 2: + Activity", ["Time", "Tcore", "Activity", "Sex_Encoded"]),
        ("Model 3: + RQ (Full)", ["Time", "Tcore", "Activity", "HeartRate", "Sex_Encoded"])
    ]

    results = []

    print("Starting Random Forest Ablation Study...\n")

    for model_name, base_feats in models:
        # 1. 自动构建特征列表 (加入 Lag 和 Delta)
        features = [f for f in base_feats if f in df.columns]

        if USE_LAG_FEATURES:
            for core_feat in ["Tcore", "Activity", "HeartRate"]:
                if core_feat in features:
                    features.append(f"{core_feat}_lag1")
                    features.append(f"d{core_feat}")

        # 过滤掉数据中没有的列
        features = [f for f in features if f in df.columns]

        # 2. 准备训练数据 (Non-torpor)
        non = df[df["Condition"] == "nontorpor"].copy()
        non = non.dropna(subset=["VO2"] + features).copy()

        if non.empty: continue

        X_non = non[features].copy()
        y_non = non["VO2"].values
        groups = non["Subject_ID"].values

        # 3. 交叉验证 (LOSO CV) 计算 RMSE
        logo = LeaveOneGroupOut()
        fold_rmses = []

        for tr, te in logo.split(X_non, y_non, groups=groups):
            # 样本权重平衡
            train_fold = non.iloc[tr]
            test_fold = non.iloc[te]

            counts = train_fold["Subject_ID"].value_counts()
            wtr = train_fold["Subject_ID"].map(lambda g: 1.0 / counts[g]).values

            rf = RandomForestRegressor(**RF_PARAMS)
            rf.fit(train_fold[features], train_fold["VO2"], sample_weight=wtr)

            pred = rf.predict(test_fold[features])
            fold_rmses.append(rmse(test_fold["VO2"], pred))

        cv_rmse = np.mean(fold_rmses)

        # 4. 全量训练并预测 Torpor
        counts = non["Subject_ID"].value_counts()
        wtr = non["Subject_ID"].map(lambda g: 1.0 / counts[g]).values

        rf_full = RandomForestRegressor(**RF_PARAMS)
        rf_full.fit(X_non, y_non, sample_weight=wtr)

        # 预测 Torpor
        tor = df[df["Condition"] == "torpor_like"].copy()
        tor = tor.dropna(subset=["VO2"] + features).copy()

        if not tor.empty:
            # Clipping (重要!)
            if CLIP_TORPOR_TO_TRAIN_RANGE:
                clip_cols = [c for c in features if c != "Sex_Encoded"]
                tor_clipped = clip_to_train_range(tor, non, clip_cols)
                X_tor = tor_clipped[features]
            else:
                X_tor = tor[features]

            pred_tor = rf_full.predict(X_tor)

            tor_rmse = rmse(tor["VO2"], pred_tor)
            avg_pred = np.mean(pred_tor)
            avg_real = np.mean(tor["VO2"])
            saving = (1 - avg_real / avg_pred) * 100
        else:
            avg_pred, avg_real, saving = np.nan, np.nan, np.nan

        results.append({
            "Model": model_name,
            "CV RMSE (Non-torpor)": cv_rmse,
            "Predicted VO2": avg_pred,
            "Real VO2": avg_real,
            "Metabolic Saving %": saving
        })

        print(f"Finished {model_name}: Saving = {saving:.2f}%")

    # 5. 输出与画图
    res_df = pd.DataFrame(results)
    print("\n=== Random Forest Ablation Results ===")
    print(res_df.to_string(index=False))

    res_df.to_csv(os.path.join(OUTDIR, "rf_ablation_results.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x="Model", y="Metabolic Saving %", palette="viridis")
    plt.title("RF Ablation Study: Which model reveals the suppression?")
    plt.ylabel("Metabolic Saving %")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "rf_ablation_plot.png"))
    plt.show()


if __name__ == "__main__":
    df = build_master(XLSX_PATH)
    run_ablation_rf(df)