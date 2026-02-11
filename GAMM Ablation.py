import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import warnings

# 忽略收敛警告
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# ============================================================
# 0) 配置 & 数据读取
# ============================================================
XLSX_PATH = "Organized data1.xlsx"

# ✅ 修改点 1: 只保留你想要的三个生理特征
FEATURE_GROUPS = {
    "Tcore": ["Tcore"],
    "Activity": ["Activity"],
    "RQ": ["HeartRate", "RQ"]
    # Time 和 Sex 已被移除，不会出现在图上
}


# ============================================================
# 1) 数据加载 & 预处理
# ============================================================
def load_and_prep_data(xlsx_path):
    # ⚠️ 请替换为你的真实读取代码 (build_dataset)
    # 这里用模拟数据演示流程
    print("⚠️ 警告: 正在使用模拟数据。请替换为真实数据读取逻辑！")
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'VO2': np.random.normal(3, 0.5, n),
        'Tcore': np.random.normal(37, 1, n),
        'Activity': np.random.exponential(1, n),
        'RQ': np.random.normal(0.8, 0.05, n),
        'Time': np.linspace(0, 100, n),
        'mouse_id': np.random.choice(['M1', 'M2', 'M3'], n),
        'condition': 'non',
        'gender_code': np.random.randint(0, 2, n)
    })

    df = df.sort_values(['mouse_id', 'Time'])
    grp = df.groupby('mouse_id')
    for col in ['Tcore', 'Activity', 'RQ']:
        df[f'{col}_lag1'] = grp[col].shift(1)
        df[f'd{col}'] = df[col] - df[f'{col}_lag1']

    df = df[df['condition'] == 'non'].dropna().copy()
    return df


# ============================================================
# 2) 构造公式
# ============================================================
def get_formula(features_to_exclude=[]):
    # 基础特征池 (依然包含 Time 和 Sex 作为背景控制变量)
    # 如果你想彻底不把 Time/Sex 放进模型，把它们注释掉即可
    base_terms = {
        "Tcore": "cr(Tcore, df=3) + cr(Tcore_lag1, df=3) + dTcore",
        "Activity": "cr(Activity, df=3) + cr(Activity_lag1, df=3) + dActivity",
        "RQ": "cr(RQ, df=3) + cr(RQ_lag1, df=3) + dRQ",
        "Time": "Time",  # 保持在模型里，但不做消融
        "Sex": "gender_code"  # 保持在模型里，但不做消融
    }

    selected_terms = []
    for group_name, term_str in base_terms.items():
        # 检查是否在排除列表中
        is_excluded = False
        for excl in features_to_exclude:
            if excl in group_name:
                is_excluded = True
                break

        if not is_excluded:
            selected_terms.append(term_str)

    if not selected_terms: return "VO2 ~ 1"
    return "VO2 ~ " + " + ".join(selected_terms)


def train_and_eval(df, formula):
    logo = LeaveOneGroupOut()
    groups = df['mouse_id'].values
    rmses = []

    for tr_idx, te_idx in logo.split(df, groups=groups):
        train = df.iloc[tr_idx]
        test = df.iloc[te_idx]
        try:
            model = smf.mixedlm(formula, train, groups=train["mouse_id"])
            res = model.fit(reml=False, disp=False)
            pred = res.predict(test)
            rmses.append(np.sqrt(mean_squared_error(test['VO2'], pred)))
        except:
            pass

    return np.mean(rmses)


# ============================================================
# 3) 主程序
# ============================================================
def run_ablation_study():
    # df = load_and_prep_data(XLSX_PATH) # <--- 换成你的真实数据
    df = load_and_prep_data(XLSX_PATH)

    print(f"Data ready: {len(df)} samples.")

    # 1. 训练 Full Model
    print("Training Full Model...", end="")
    full_formula = get_formula(features_to_exclude=[])
    full_rmse = train_and_eval(df, full_formula)
    print(f" RMSE = {full_rmse:.4f}")

    results = []

    # 2. 训练消融模型 (只循环 Tcore, Activity, RQ)
    for group_name, keywords in FEATURE_GROUPS.items():
        print(f"Training model without '{group_name}'...", end="")
        reduced_formula = get_formula(features_to_exclude=[group_name])
        rmse = train_and_eval(df, reduced_formula)

        diff = rmse - full_rmse
        pct_diff = (diff / full_rmse) * 100

        print(f" RMSE = {rmse:.4f} (+{diff:.4f})")

        results.append({
            "Feature": group_name,
            "Increase": diff,
            "PctIncrease": pct_diff
        })

    # 3. 画图
    res_df = pd.DataFrame(results).sort_values("Increase", ascending=True)

    plt.figure(figsize=(8, 5))  # 稍微调小一点，因为条子少了

    # 颜色越深代表越重要
    colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(res_df)))

    bars = plt.barh(res_df["Feature"], res_df["Increase"], color=colors, edgecolor='black', alpha=0.8)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001,
                 bar.get_y() + bar.get_height() / 2,
                 f'+{width:.3f}\n({res_df.loc[res_df["Increase"] == width, "PctIncrease"].values[0]:.1f}%)',
                 va='center', fontsize=11, color='black', fontweight='bold')

    plt.axvline(0, color='black', linewidth=1)

    plt.title("Physiological Feature Importance (Ablation Study)", fontsize=14, fontweight='bold')
    plt.xlabel("Increase in RMSE (Loss of Predictive Power)", fontsize=12)
    plt.ylabel("Physiological Predictor", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig("gamm_ablation_3features.png", dpi=300)
    print("✅ Done! Saved to gamm_ablation_3features.png")
    plt.show()


if __name__ == "__main__":
    run_ablation_study()