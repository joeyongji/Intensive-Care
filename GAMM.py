import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略不必要的收敛警告
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

from sklearn.metrics import mean_squared_error
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix, build_design_matrices

# ============================================================
# 0) 配置 (Config)
# ============================================================
# 请确保这里的文件名和你上传的一致
PATH = "Organized data1.xlsx"

# Sheet 映射关系
SHEETS = {
    ("torpor", "Tcore"): "Tcore torpor-like",
    ("non", "Tcore"): "Tcore nontorpor",
    ("torpor", "VO2"): "VO2 torpor-like",
    ("non", "VO2"): "VO2 nontorpor",
    ("torpor", "Activity"): "activity torpor-like",
    ("non", "Activity"): "ativity nontorpor",
    ("torpor", "RQ"): "Heart Rate torpor-like",  # 注意：这里实际上读的是 RQ 数据
    ("non", "RQ"): "Heart Rate nontorpor",
}

# 参与建模的特征列表
FULL_FEATURES = ["Tcore", "Activity", "RQ"]

# 模型参数
LAGS = [1, 2, 3]  # 滞后步数
USE_DELTA = True  # 是否使用差分
TIME_RANGE = (1, 119)  # 时间范围筛选
CLIP_TORPOR_TO_NON = True  # 防止 Torpor 测试数据超出 Non 训练数据的范围
SPLINE_DF = 5  # 样条自由度 (控制曲线灵活性)


# ============================================================
# 1) 数据读取与清洗函数
# ============================================================
def read_wide_sheet(sheet_name: str):
    """读取宽表并转换为长表，尝试自动提取性别"""
    try:
        df = pd.read_excel(PATH, sheet_name=sheet_name)
    except ValueError:
        print(f"❌ Error: Sheet '{sheet_name}' not found.")
        return None, None

    time_col = df.columns[0]

    # 尝试从第一行提取性别 (M/F)
    first_row = df.iloc[0]
    if first_row.astype(str).str.contains('M|F', regex=True).any():
        mouse_cols = [c for c in df.columns[1:]]
        gender_map = {str(c): str(first_row[c]).strip() for c in mouse_cols}
        df_num = df.iloc[1:].copy()
    else:
        # 如果第一行没有，尝试找包含空值的行作为 metadata
        gender_row = df[df[time_col].isna()]
        if not gender_row.empty:
            gender_row = gender_row.iloc[0]
            mouse_cols = [c for c in df.columns[1:]]
            gender_map = {str(c): str(gender_row[c]).strip() for c in mouse_cols}
            df_num = df[df[time_col].notna()].copy()
        else:
            gender_map = None
            df_num = df.copy()

    df_num = df_num.rename(columns={time_col: "time"})
    long = df_num.melt(id_vars=["time"], var_name="mouse_id", value_name="value")
    long["mouse_id"] = long["mouse_id"].astype(str)
    return long, gender_map


def build_dataset():
    """遍历所有 Sheet 构建总表"""
    parts = []
    gender_master = {}

    for (cond, var), sh in SHEETS.items():
        long, gmap = read_wide_sheet(sh)
        if long is None: continue
        if gmap:
            for k, v in gmap.items():
                if k not in gender_master: gender_master[k] = v
        long["condition"] = cond
        long["var"] = var
        parts.append(long)

    if not parts: return pd.DataFrame()

    df_all = pd.concat(parts, ignore_index=True)
    # 转回宽表格式: [condition, time, mouse_id] 为主键，var 为列
    df_wide = df_all.pivot_table(index=["condition", "time", "mouse_id"],
                                 columns="var", values="value", aggfunc="mean").reset_index()
    df_wide["gender"] = df_wide["mouse_id"].map(gender_master)
    return df_wide


def clip_to_train_range(df_in: pd.DataFrame, train_ref: pd.DataFrame, cols):
    """防止 Spline 外推报错，将测试集特征限制在训练集范围内"""
    df_out = df_in.copy()
    for c in cols:
        if c in df_out.columns and c in train_ref.columns:
            lo, hi = train_ref[c].min(), train_ref[c].max()
            df_out[c] = df_out[c].clip(lo, hi)
    return df_out


# ============================================================
# 2) 特征工程与建模核心
# ============================================================
def prepare_features(df_raw, active_features):
    """根据 active_features 生成 lags 和 delta"""
    df = df_raw.copy()

    # 1. 确保数值类型
    for c in ["VO2"] + FULL_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["condition", "mouse_id", "time"])

    # 2. 生成 Lag 和 Delta
    for col in active_features:
        for L in LAGS:
            df[f"{col}_lag{L}"] = df.groupby(["condition", "mouse_id"])[col].shift(L)
        if USE_DELTA:
            df[f"{col}_delta1"] = df[col] - df.groupby(["condition", "mouse_id"])[col].shift(1)

    # 3. 删除因 Lag 产生的空值行
    cols_to_check = []
    for col in active_features:
        cols_to_check.append(col)
        cols_to_check.extend([f"{col}_lag{L}" for L in LAGS])
        if USE_DELTA: cols_to_check.append(f"{col}_delta1")

    df = df.dropna(subset=cols_to_check + ["VO2", "gender", "mouse_id"]).copy()

    # 4. 性别编码
    if "gender" in df.columns and df["gender"].notna().any():
        df["gender_code"] = (df["gender"].astype(str).str.upper().str.contains("F")).astype(int)
    else:
        df["gender_code"] = 0

    return df


def make_spline_design(df, spline_cols):
    """构建 B-Spline 设计矩阵"""
    if not spline_cols:
        formula = "gender_code + time"
    else:
        terms = [f"bs({c}, df={SPLINE_DF}, degree=3, include_intercept=False)" for c in spline_cols]
        formula = " + ".join(terms) + " + gender_code + time"

    X = dmatrix(formula, df, return_type="dataframe")
    return X, X.design_info


def run_gamm_ablation(train_df, test_df, active_feats, model_name):
    """训练单个模型并评估"""
    # 1. 确定所有相关列 (当前值 + lag + delta)
    spline_cols = active_feats + [f"{c}_lag{L}" for c in active_feats for L in LAGS]
    if USE_DELTA:
        spline_cols += [f"{c}_delta1" for c in active_feats]

    # 2. 构建训练集矩阵
    X_train, design_info = make_spline_design(train_df, spline_cols)
    y_train = train_df["VO2"].values
    g_train = train_df["mouse_id"].values

    # 3. 训练 MixedLM
    print(f"   Training {model_name} with features: {active_feats}...")
    try:
        model = MixedLM(endog=y_train, exog=X_train, groups=g_train)
        # 优先用 lbfgs，如果失败尝试 powell
        res = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
        if not res.converged:
            print("     ⚠️ lbfgs not converged, trying powell...")
            res = model.fit(reml=True, method="powell", maxiter=1000, disp=False)

    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return None

    # 4. 预测测试集 (Torpor)
    test_eval = test_df.copy()
    if CLIP_TORPOR_TO_NON:
        test_eval = clip_to_train_range(test_eval, train_df, spline_cols)

    X_test = build_design_matrices([design_info], test_eval, return_type="dataframe")[0]
    pred = res.predict(exog=X_test)

    # 5. 计算指标
    real = test_eval["VO2"].values
    rmse = np.sqrt(mean_squared_error(real, pred))

    # 仅用于记录，不用于画图
    avg_real = np.mean(real)
    avg_pred = np.mean(pred)

    return {
        "Model": model_name,
        "Removed Feature": list(set(FULL_FEATURES) - set(active_feats))[0] if len(active_feats) < len(
            FULL_FEATURES) else "None",
        "RMSE": rmse,
        "Pred_Mean": avg_pred,  # 留着 debug 用
        "Real_Mean": avg_real
    }


# ============================================================
# 3) 主程序：执行消融循环 (LOFO)
# ============================================================
if __name__ == "__main__":
    print("🚀 Starting Data Processing...")
    data_raw = build_dataset()

    if data_raw.empty:
        print("❌ Data load failed. Check path.")
    else:
        # 筛选时间范围
        data_raw = data_raw[(data_raw["time"] >= TIME_RANGE[0]) & (data_raw["time"] <= TIME_RANGE[1])].copy()

        results = []

        # --- Step 1: Full Model (基准) ---
        print("\n🔹 Running Full Model (Reference)...")
        df_full = prepare_features(data_raw, FULL_FEATURES)
        tr_full = df_full[df_full["condition"] == "non"]
        te_full = df_full[df_full["condition"] == "torpor"]

        res_full = run_gamm_ablation(tr_full, te_full, FULL_FEATURES, "Full Model")
        if res_full: results.append(res_full)

        # --- Step 2: Ablation Loop (逐个移除) ---
        print("\n🔹 Running Leave-One-Feature-Out Ablation...")
        for feature_to_remove in FULL_FEATURES:
            current_feats = [f for f in FULL_FEATURES if f != feature_to_remove]
            model_name = f"Minus {feature_to_remove}"

            # 重新生成数据 (因为不同特征组合 dropna 结果不同)
            df_curr = prepare_features(data_raw, current_feats)
            tr_curr = df_curr[df_curr["condition"] == "non"]
            te_curr = df_curr[df_curr["condition"] == "torpor"]

            res = run_gamm_ablation(tr_curr, te_curr, current_feats, model_name)
            if res: results.append(res)

        # --- Step 3: 结果展示与绘图 ---
        if results:
            res_df = pd.DataFrame(results)

            # 计算 RMSE Impact (相对于 Full Model 增加了多少错误)
            full_row = res_df[res_df["Model"] == "Full Model"]
            if not full_row.empty:
                full_rmse = full_row["RMSE"].values[0]
                res_df["RMSE Impact"] = res_df["RMSE"] - full_rmse
            else:
                res_df["RMSE Impact"] = np.nan

            print("\n" + "=" * 70)
            print("             GAMM LOFO ABLATION RESULTS (RMSE IMPACT)             ")
            print("=" * 70)
            # 打印表格供分析
            cols = ["Model", "Removed Feature", "RMSE", "RMSE Impact", "Pred_Mean"]
            print(res_df[cols].to_string(index=False))
            print("=" * 70)

            # --- 只画 RMSE Impact 图 ---
            # 过滤掉 Full Model (因为它的 Impact 是 0，没必要画，或者画出来也是基准线)
            plot_df = res_df[res_df["Model"] != "Full Model"].copy()

            # 设置绘图风格
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(8, 6))

            # 画柱状图
            ax = sns.barplot(data=plot_df, x="Removed Feature", y="RMSE Impact", palette="Reds")

            # 添加标题和标签
            plt.title("Feature Importance Analysis (GAMM)\n(Higher bar = Feature is more important)", fontsize=14)
            plt.ylabel("Increase in RMSE (Loss of Accuracy)", fontsize=12)
            plt.xlabel("Removed Feature", fontsize=12)
            plt.axhline(0, color='black', linewidth=1)  # 基准线

            # 在柱子上标数值
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'+{p.get_height():.3f}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 5), textcoords='offset points')

            plt.tight_layout()
            plt.show()

            print("✅ Analysis Complete. Chart generated.")
        else:
            print("❌ No valid results generated.")