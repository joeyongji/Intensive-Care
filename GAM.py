import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygam import LinearGAM, s, f
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. 配置路径和 Sheet 映射
# ==========================================
PATH = "Organized data.xlsx"  # 请确保文件名正确

# 你的 Sheet 映射表
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
    """
    读宽表 -> 提取性别 -> 转长表
    """
    try:
        df = pd.read_excel(PATH, sheet_name=sheet_name)
    except ValueError:
        print(f"❌ Error: Sheet '{sheet_name}' not found.")
        return None, None

    # 第一列通常是 time (min)
    time_col = df.columns[0]

    # 提取性别行 (time 为 NaN 或 第一行包含 M/F)
    # 先看第一行是否是 M/F
    first_row = df.iloc[0]
    if first_row.astype(str).str.contains('M|F').any():
        mouse_cols = [c for c in df.columns[1:]]  # 第一列是时间，后面是老鼠
        gender_map = {c: str(first_row[c]).strip() for c in mouse_cols}
        df_num = df.iloc[1:].copy()  # 去掉性别行
    else:
        # 如果不是第一行，尝试找 NaN 行 (你原来的逻辑)
        gender_row = df[df[time_col].isna()]
        if not gender_row.empty:
            gender_row = gender_row.iloc[0]
            mouse_cols = [c for c in df.columns[1:]]
            gender_map = {c: str(gender_row[c]).strip() for c in mouse_cols}
            df_num = df[df[time_col].notna()].copy()
        else:
            print(f"⚠️ Warning: No gender info in {sheet_name}")
            gender_map = None
            df_num = df.copy()

    df_num.rename(columns={time_col: "time"}, inplace=True)

    # 宽表转长表
    # 注意：mouse_id 可能是字符串 '1', '2' 或数字 1, 2，统一转 string 处理防止报错
    long = df_num.melt(id_vars=["time"], var_name="mouse_id", value_name="value")

    return long, gender_map


def build_long_dataset():
    parts = []
    gender_master = {}  # 汇总所有老鼠的性别

    for (cond, var), sh in SHEETS.items():
        long, gmap = read_wide_sheet(sh)
        if long is None: continue

        # 更新总性别表
        if gmap:
            for k, v in gmap.items():
                k_str = str(k)  # 统一转字符串 key
                if k_str not in gender_master:
                    gender_master[k_str] = v

        long["condition"] = cond
        long["var"] = var
        long["mouse_id"] = long["mouse_id"].astype(str)  # 统一 ID 格式
        parts.append(long)

    df_all = pd.concat(parts, ignore_index=True)

    # Pivot: 把变量变成列 (Tcore, VO2, ...)
    df_wide = df_all.pivot_table(index=["condition", "time", "mouse_id"],
                                 columns="var", values="value", aggfunc="mean").reset_index()

    # 映射性别
    df_wide["gender"] = df_wide["mouse_id"].map(gender_master)

    return df_wide


# ==========================================
# 3. 构建数据集 & 特征工程 (Time Lag)
# ==========================================
data = build_long_dataset()

# 筛选时间范围
data = data[(data["time"] >= 1) & (data["time"] <= 119)].copy()

# 确保数值列也是数字
cols_to_numeric = ["VO2", "Tcore", "Activity", "RQ"]
for c in cols_to_numeric:
    data[c] = pd.to_numeric(data[c], errors='coerce')

# 去除空值
needed = cols_to_numeric + ["gender", "mouse_id"]
df = data.dropna(subset=needed).copy()

# 编码性别: F=1, M=0
df["gender_code"] = (df["gender"].str.upper().str.contains("F")).astype(int)

# --- 关键修改：加入 Time Lag (上一分钟的 VO2) ---
# 先按 mouse_id 和 time 排序
df = df.sort_values(by=["condition", "mouse_id", "time"])
# 这行代码计算上一分钟的 VO2
df["VO2_lag1"] = df.groupby(["condition", "mouse_id"])["VO2"].shift(1)

# 因为第一分钟没有上一分钟的数据，会产生 NaN，需要再次 dropna
df = df.dropna(subset=["VO2_lag1"]).copy()

# 拆分训练集 (Non-torpor) 和 测试集 (Torpor)
train = df[df["condition"] == "non"].copy()
test_torpor = df[df["condition"] == "torpor"].copy()

print(f"Train size: {train.shape}, Test size: {test_torpor.shape}")

# ==========================================
# 4. GAM 模型定义与训练
# ==========================================
# X: [VO2_lag1, Tcore, Activity, RQ, gender_code]
# 注意顺序：s(0) 对应第0列，s(1) 对应第1列...

X_train = train[["VO2_lag1", "Tcore", "Activity", "RQ", "gender_code"]].values
y_train = train["VO2"].values

X_torpor = test_torpor[["VO2_lag1", "Tcore", "Activity", "RQ", "gender_code"]].values
y_torpor = test_torpor["VO2"].values

# 定义 GAM
# s(0): VO2_lag1 (时间惯性，非常重要)
# s(1): Tcore
# s(2): Activity
# s(3): RQ
# f(4): Gender (分类变量)
# 注意：这里移除了 mouse_id，提高泛化能力
gam = LinearGAM(
    s(0, n_splines=10) +
    s(1, n_splines=10) +
    s(2, n_splines=10) +
    s(3, n_splines=10) +
    f(4)
)

print("\n正在进行 Grid Search 寻找最优参数...")
gam.gridsearch(X_train, y_train)
print(gam.summary())


# ==========================================
# 5. 预测与评估 (Clipping Trick)
# ==========================================
# 这里的 Clipping 是为了防止外推。
# 我们把 Torpor 组的特征限制在 Non-torpor 组的范围内。

def clip_to_train_range(df_in, train_ref, cols):
    df_out = df_in.copy()
    for c in cols:
        lo, hi = train_ref[c].min(), train_ref[c].max()
        df_out[c] = df_out[c].clip(lo, hi)
    return df_out


# 即使加了 lag1，体温等其他变量依然需要 clip 以防万一
clip_cols = ["VO2_lag1", "Tcore", "Activity", "RQ"]
test_torpor_clip = clip_to_train_range(test_torpor, train, clip_cols)

# 准备预测数据 X
X_test_clip = test_torpor_clip[["VO2_lag1", "Tcore", "Activity", "RQ", "gender_code"]].values

# 进行预测
test_torpor["VO2_pred"] = gam.predict(X_test_clip)

# 计算误差
mse = mean_squared_error(test_torpor["VO2"], test_torpor["VO2_pred"])
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_torpor["VO2"], test_torpor["VO2_pred"])

print(f"\nTorpor-like Prediction RMSE: {rmse:.4f}")
print(f"Torpor-like Prediction MAE : {mae:.4f}")

# ==========================================
# 6. 可视化：四条曲线对比
# ==========================================
# 为了画图，也给 Train 集生成预测值
train["VO2_pred"] = gam.predict(X_train)

# 计算每分钟的平均值 (Time-level Mean)
mean_non = train.groupby("time")[["VO2", "VO2_pred"]].mean().reset_index()
mean_tor = test_torpor.groupby("time")[["VO2", "VO2_pred"]].mean().reset_index()
plt.figure(figsize=(10, 6))

# 使用四种不同的颜色
colors = ['blue', 'orange', 'green', 'purple']  # 定义四种颜色

# Non-torpor: 实线=真实，虚线=预测
plt.plot(mean_non["time"], mean_non["VO2"], color=colors[0], linestyle='-',
         label="Non-torpor Actual", alpha=0.6)
plt.plot(mean_non["time"], mean_non["VO2_pred"], color=colors[1], linestyle='--',
         label="Non-torpor Predicted")

# Torpor: 实线=真实，虚线=预测(基于Non-torpor模型)
plt.plot(mean_tor["time"], mean_tor["VO2"], color=colors[2], linestyle='-',
         label="Torpor Actual", alpha=0.8)
plt.plot(mean_tor["time"], mean_tor["VO2_pred"], color=colors[3], linestyle='--',
         label="Torpor Predicted (by Non-torpor Model)")

plt.xlabel("Time (min)")
plt.ylabel("VO2 (Metabolic Rate)")
plt.title("GAM Model: Can Normal Physiology Explain Torpor?")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
