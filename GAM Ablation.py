import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pygam import LinearGAM, s, f
from sklearn.metrics import mean_squared_error

# =================================================================
# 1. 配置与数据读取
# =================================================================
EXCEL_FILENAME = "Organized data1.xlsx"  # 确保文件名正确

config_map = {
    "Tcore torpor": ("Tcore", "Torpor"),
    "Tcore nontorpor": ("Tcore", "Non-torpor"),
    "VO2 torpor": ("VO2", "Torpor"),
    "VO2 nontorpor": ("VO2", "Non-torpor"),
    "activity torpor": ("Activity", "Torpor"),
    "ativity nontorpor": ("Activity", "Non-torpor"),
    "Heart Rate torpor": ("RQ", "Torpor"),  # 这里把 HeartRate 当作 RQ
    "Heart Rate nontorpor": ("RQ", "Non-torpor")
}


def load_data_robust(filename, config_map):
    if not os.path.exists(filename):
        print(f"❌ 错误：找不到文件 {filename}")
        return None

    print("正在读取数据...")
    xls = pd.ExcelFile(filename)
    all_dfs = []

    for sheet_name in xls.sheet_names:
        for key, (feat, grp) in config_map.items():
            if key in sheet_name:
                df = pd.read_excel(xls, sheet_name=sheet_name)

                # 找时间列
                time_col = None
                for c in df.columns:
                    if 'time' in str(c).lower():
                        time_col = c
                        break
                if time_col is None: continue
                df = df.rename(columns={time_col: 'Time'})

                # 提取性别信息 (如果第一行是 M/F)
                # 我们简单处理：如果第一行是性别，就把它删掉，但为了简单起见，这里先忽略性别特征的影响
                if not df.empty and any(x in ['M', 'F'] for x in df.iloc[0].astype(str).values):
                    df = df.drop(0).reset_index(drop=True)

                # 重命名 Subject 列
                subj_cols = [c for c in df.columns if c != 'Time' and 'Unnamed' not in str(c)]
                rename_dict = {old: f"Mouse_{i + 1:02d}" for i, old in enumerate(subj_cols)}
                df = df.rename(columns=rename_dict)

                # 转长表
                df_long = df.melt(id_vars=['Time'], value_vars=list(rename_dict.values()),
                                  var_name='Subject_ID', value_name=feat)
                df_long['Group'] = grp
                df_long[feat] = pd.to_numeric(df_long[feat], errors='coerce')
                all_dfs.append(df_long)
                break

    if not all_dfs: return None
    # 合并
    return pd.concat(all_dfs, ignore_index=True).groupby(['Time', 'Group', 'Subject_ID']).first().reset_index()


# =================================================================
# 2. 数据预处理 (生成 Lag 特征 & 划分数据集)
# =================================================================
df = load_data_robust(EXCEL_FILENAME, config_map)

if df is None:
    exit()

print("正在进行特征工程 (生成 Lag)...")
# 确保数值类型
cols = ['Tcore', 'RQ', 'Activity', 'VO2']
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 排序以计算 Lag (按老鼠、时间排序)
df = df.sort_values(by=['Group', 'Subject_ID', 'Time'])

# 计算 Lag (上一分钟的 VO2)
df['VO2_lag1'] = df.groupby(['Group', 'Subject_ID'])['VO2'].shift(1)

# 删除因为 shift 产生的第一行空值，以及其他的空值
dropna_cols = cols + ['VO2_lag1']
df_clean = df.dropna(subset=dropna_cols).copy()

# 为了代码兼容，增加一个 dummy gender_code (全为0，如果不研究性别差异的话)
df_clean['gender_code'] = 0

# 划分训练集 (Non-torpor) 和 测试集 (Torpor)
train = df_clean[df_clean['Group'] == 'Non-torpor'].copy()
test_torpor = df_clean[df_clean['Group'] == 'Torpor'].copy()

print(f"训练集样本数: {len(train)}")
print(f"Torpor 测试集样本数: {len(test_torpor)}")

# =================================================================
# 3. GAM 消融实验 (Ablation Loop)
# =================================================================
# 定义模型配置
models_config = [
    {
        "name": "Model 1: Dynamic Baseline",
        "features": ["VO2_lag1", "Tcore"],
        "formula": s(0, n_splines=10) + s(1, n_splines=10),
        "desc": "History + Tcore (Physics)"
    },
    {
        "name": "Model 2: + Activity",
        "features": ["VO2_lag1", "Tcore", "Activity"],
        "formula": s(0) + s(1) + s(2),
        "desc": "+ Behavioral State"
    },
    {
        "name": "Model 3: + RQ (Full)",
        "features": ["VO2_lag1", "Tcore", "Activity", "RQ"],
        "formula": s(0) + s(1) + s(2) + s(3),
        "desc": "+ Metabolic Fuel"
    }
]

results = []

print("\n=== 开始 GAM 消融实验 ===")

for config in models_config:
    print(f"正在训练 {config['name']}...")

    feats = config["features"]

    # 准备矩阵 (PyGAM 需要 numpy array)
    X_train = train[feats].values
    y_train = train["VO2"].values

    X_test = test_torpor[feats].values
    y_real = test_torpor["VO2"].values

    # 训练 GAM
    gam = LinearGAM(config["formula"])
    gam.gridsearch(X_train, y_train)

    # 预测 (带简单的 Clipping 防止外推)
    X_test_clip = X_test.copy()
    for i, col_name in enumerate(feats):
        lo, hi = train[col_name].min(), train[col_name].max()
        X_test_clip[:, i] = np.clip(X_test_clip[:, i], lo, hi)

    pred = gam.predict(X_test_clip)

    # 计算指标
    avg_pred = pred.mean()
    avg_real = y_real.mean()
    gap_pct = (1 - avg_real / avg_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_real, pred))

    results.append({
        "Model": config["name"],
        "Description": config["desc"],
        "RMSE": rmse,
        "Predicted_VO2": avg_pred,
        "Real_VO2": avg_real,
        "Metabolic Saving %": gap_pct
    })

# =================================================================
# 4. 生成报告与图表
# =================================================================
res_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("             GAM ABLATION RESULTS (消融实验结果)            ")
print("=" * 60)
print(res_df[['Model', 'Description', 'Predicted_VO2', 'Real_VO2', 'Metabolic Saving %']].to_string(index=False))
print("=" * 60)

# 画图
plt.figure(figsize=(10, 6))
# 使用柱状图展示 Saving %
barplot = sns.barplot(data=res_df, x='Model', y='Metabolic Saving %', palette='viridis')

# 在柱子上标数值
for p in barplot.patches:
    barplot.annotate(f'{p.get_height():.2f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),
                     textcoords='offset points')

plt.title('Ablation Study: Which Model Reveals the Largest Metabolic Gap?', fontsize=14)
plt.ylabel('Metabolic Suppression (%)', fontsize=12)
plt.xlabel('Model Complexity', fontsize=12)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('GAM_Ablation_Result.png')
plt.show()