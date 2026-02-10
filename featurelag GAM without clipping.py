import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygam import LinearGAM, s, f
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. 配置路径和 Sheet 映射
# ==========================================
PATH = "Organized data.xlsx"

# 请确保这里的 Sheet 名字和你 Excel 里的一致
SHEETS = {
    ("torpor", "Tcore"): "Tcore torpor-like",
    ("non", "Tcore"): "Tcore nontorpor",
    ("torpor", "VO2"): "VO2 torpor-like",
    ("non", "VO2"): "VO2 nontorpor",
    ("torpor", "Activity"): "activity torpor-like",
    ("non", "Activity"): "ativity nontorpor",
    ("torpor", "RQ"): "Heart Rate torpor-like",  # RQ 数据
    ("non", "RQ"): "Heart Rate nontorpor",
}


# ==========================================
# 2. 数据读取与清洗函数
# ==========================================
def read_wide_sheet(sheet_name):
    try:
        df = pd.read_excel(PATH, sheet_name=sheet_name)
    except ValueError:
        print(f"❌ Error: Sheet '{sheet_name}' not found.")
        return None, None

    # 第一列通常是 time (min)
    time_col = df.columns[0]

    # 提取性别行
    first_row = df.iloc[0]
    if first_row.astype(str).str.contains('M|F').any():
        mouse_cols = [c for c in df.columns[1:]]
        gender_map = {c: str(first_row[c]).strip() for c in mouse_cols}
        df_num = df.iloc[1:].copy()
    else:
        gender_row = df[df[time_col].isna()]
        if not gender_row.empty:
            gender_row = gender_row.iloc[0]
            mouse_cols = [c for c in df.columns[1:]]
            gender_map = {c: str(gender_row[c]).strip() for c in mouse_cols}
            df_num = df[df[time_col].notna()].copy()
        else:
            gender_map = None
            df_num = df.copy()

    df_num.rename(columns={time_col: "time"}, inplace=True)

    # 宽表转长表
    long = df_num.melt(id_vars=["time"], var_name="mouse_id", value_name="value")
    return long, gender_map


def build_long_dataset():
    parts = []
    gender_master = {}

    for (cond, var), sh in SHEETS.items():
        long, gmap = read_wide_sheet(sh)
        if long is None: continue

        if gmap:
            for k, v in gmap.items():
                k_str = str(k)
                if k_str not in gender_master:
                    gender_master[k_str] = v

        long["condition"] = cond
        long["var"] = var
        long["mouse_id"] = long["mouse_id"].astype(str)
        parts.append(long)

    df_all = pd.concat(parts, ignore_index=True)

    # Pivot: 把变量变成列
    df_wide = df_all.pivot_table(index=["condition", "time", "mouse_id"],
                                 columns="var", values="value", aggfunc="mean").reset_index()

    # 映射性别
    df_wide["gender"] = df_wide["mouse_id"].map(gender_master)
    return df_wide


# ==========================================
# 3. 构建 Feature-Lag 数据集 (关键修改)
# ==========================================
data = build_long_dataset()

# 筛选时间
data = data[(data["time"] >= 1) & (data["time"] <= 119)].copy()

# 转换数值类型
cols_to_numeric = ["VO2", "Tcore", "Activity", "RQ"]
for c in cols_to_numeric:
    data[c] = pd.to_numeric(data[c], errors='coerce')

needed = cols_to_numeric + ["gender", "mouse_id"]
df = data.dropna(subset=needed).copy()

# 编码性别
df["gender_code"] = (df["gender"].str.upper().str.contains("F")).astype(int)

# --- 核心修改：生成 Feature Lags ---
# 我们不再 Lag VO2，而是 Lag 生理特征
df = df.sort_values(by=["condition", "mouse_id", "time"])

feature_cols = ["Tcore", "Activity", "RQ"]
for col in feature_cols:
    # 计算上一分钟的值
    df[f"{col}_lag1"] = df.groupby(["condition", "mouse_id"])[col].shift(1)

# 删除因为 Shift 产生的空值 (即每只老鼠的第一分钟)
df = df.dropna().copy()

# 拆分数据集
train = df[df["condition"] == "non"].copy()
test_torpor = df[df["condition"] == "torpor"].copy()

print(f"Train size: {train.shape}, Test size: {test_torpor.shape}")

# ==========================================
# 4. GAM 模型定义 (Feature-Lag Model)
# ==========================================
# 输入特征 X 包含：
# 1. Tcore (当前)
# 2. Tcore_lag1 (上一分钟)
# 3. Activity (当前)
# 4. Activity_lag1 (上一分钟)
# 5. RQ (当前)
# 6. RQ_lag1 (上一分钟)
# 7. Gender

input_features = [
    "Tcore", "Tcore_lag1",
    "Activity", "Activity_lag1",
    "RQ", "RQ_lag1",
    "gender_code"
]

X_train = train[input_features].values
y_train = train["VO2"].values

X_torpor = test_torpor[input_features].values
y_torpor = test_torpor["VO2"].values

# 定义 GAM
# s(0)-s(5): 6个连续变量
# f(6): 性别
gam = LinearGAM(
    s(0, n_splines=10) + s(1, n_splines=10) +
    s(2, n_splines=10) + s(3, n_splines=10) +
    s(4, n_splines=10) + s(5, n_splines=10) +
    f(6)
)

print("\n正在训练 Feature-Lag GAM 模型...")
gam.gridsearch(X_train, y_train)
print(gam.summary())


# ==========================================
# 5. 预测与评估 (NO Clipping - 原始外推)
# ==========================================
# 提示：这时候我们直接把 Torpor 的极低体温喂给只见过正常体温的模型
# 模型会尝试进行"外推" (Extrapolation)

# 1. 直接构建输入 X (使用原始 test_torpor)
# input_features = ["Tcore", "Tcore_lag1", "Activity", ..., "gender_code"]
X_test_raw = test_torpor[input_features].values

# 2. 进行预测
test_torpor["VO2_pred"] = gam.predict(X_test_raw)

# 3. 误差计算
mse = mean_squared_error(test_torpor["VO2"], test_torpor["VO2_pred"])
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_torpor["VO2"], test_torpor["VO2_pred"])

print(f"\nFeature-Lag Prediction RMSE (No Clipping): {rmse:.4f}")
print(f"Feature-Lag Prediction MAE  (No Clipping): {mae:.4f}")

# ==========================================
# 6. 可视化
# ==========================================
# 设置全局样式
plt.rcParams.update({
    'font.size': 12,           # 全局字体大小（影响标题、坐标轴、图例）
    'axes.labelsize': 14,      # 坐标轴标签字体大小 (xlabel, ylabel)
    'axes.titlesize': 16,      # 标题字体大小
    'legend.fontsize': 12,     # 图例字体大小
    'lines.linewidth': 2.0,    # 线条宽度（全局默认值）
})

# 给 Train 集也生成预测值
train["VO2_pred"] = gam.predict(X_train)

# 计算每分钟的平均值 (Time-level Mean)
mean_non = train.groupby("time")[["VO2", "VO2_pred"]].mean().reset_index()
mean_tor = test_torpor.groupby("time")[["VO2", "VO2_pred"]].mean().reset_index()

plt.figure(figsize=(10, 6))

# 定义色盲友好的颜色 (Okabe-Ito Palette)
c_non = '#0072B2'
c_tor = '#D55E00'

# Non-torpor:
plt.plot(mean_non["time"], mean_non["VO2"], 'b-', label="Non-torpor (train): observed", alpha=0.6, color=c_non, linewidth=2.0)
plt.plot(mean_non["time"], mean_non["VO2_pred"], 'b--', label="Non-torpor (train): predicted", color=c_non, linewidth=2.0)

# Torpor:
plt.plot(mean_tor["time"], mean_tor["VO2"], 'r-', label="Torpor (test): observed", alpha=0.8, color=c_tor, linewidth=2.0)
plt.plot(mean_tor["time"], mean_tor["VO2_pred"], 'r--', label="Torpor (test): predicted", color=c_tor, linewidth=2.0)

plt.xlabel("Time (min)", fontsize=14)
plt.ylabel("Mean VO₂ (mL/kg)", fontsize=14)
plt.title("Feature-Lag GAM Model: mean VO₂ over time", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
