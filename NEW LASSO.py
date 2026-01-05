import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 定义 Excel 文件路径 和 子表映射
# ==========================================
excel_path = 'Organized data.xlsx'

# 确保这里的名字和你 Excel 里的 Sheet 名字完全一致
SHEET_MAP = {
    'Tcore_Torpor': 'Tcore torpor-like',
    'Tcore_NonTorpor': 'Tcore nontorpor',
    'VO2_Torpor': 'VO2 torpor-like',
    'VO2_NonTorpor': 'VO2 nontorpor',
    'Activity_Torpor': 'activity torpor-like',
    'Activity_NonTorpor': 'ativity nontorpor',
    'HR_Torpor': 'Heart Rate torpor-like',
    'HR_NonTorpor': 'Heart Rate nontorpor'
}


# ==========================================
# 2. 数据处理函数 (Excel + Gender 提取)
# ==========================================
def process_sheet_with_gender(excel_file, sheet_name, metric_name):
    """
    读取指定 Sheet，提取性别，清洗数据，并转换为 '长格式'
    """
    try:
        # 使用 read_excel 读取
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
    except ValueError:
        print(f"❌ 找不到 Sheet: {sheet_name}，请检查名称拼写！")
        return None
    except Exception as e:
        print(f"❌ 读取 {sheet_name} 时发生未知错误: {e}")
        return None

    # 1. 找到时间列
    time_col = [c for c in df.columns if 'time' in str(c).lower()]
    if not time_col:
        print(f"⚠️ 在 {sheet_name} 中没找到 'time' 列，跳过。")
        return None
    time_col = time_col[0]

    # -----------------------------------------------
    # 核心修改：提取性别信息 (Gender Extraction)
    # -----------------------------------------------
    # 假设第0行包含性别 (M/F)
    first_row = df.iloc[0]

    # 检查第一行是否包含 M 或 F
    if first_row.astype(str).str.contains('M|F').any():
        # 创建映射字典：列名(小鼠ID) -> 性别数值 (0=Male, 1=Female)
        gender_map = {}
        for col in df.columns:
            if col == time_col: continue  # 跳过时间列

            val = str(first_row[col]).upper().strip()
            if 'M' in val:
                gender_map[col] = 0  # Male
            elif 'F' in val:
                gender_map[col] = 1  # Female
            else:
                gender_map[col] = np.nan  # 未知

        # 提取完性别后，删除这一行
        df = df.iloc[1:]
    else:
        print(f"⚠️ 警告: 在 {sheet_name} 中没找到性别行 (M/F)，将不包含性别特征。")
        gender_map = None

    # 2. 转换时间为数字，并设为索引
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.set_index(time_col)

    # 3. "熔化" (Melt)：把宽表变成长表
    df_long = df.stack().reset_index()
    df_long.columns = ['Time', 'MouseID', metric_name]

    # 4. 把性别映射回去
    if gender_map:
        df_long['Gender'] = df_long['MouseID'].map(gender_map)
    else:
        df_long['Gender'] = 0  # 默认全为 0 (如果没找到性别)

    # 5. 转为数值型
    df_long[metric_name] = pd.to_numeric(df_long[metric_name], errors='coerce')

    # 6. 筛选时间 > 15分钟
    df_long = df_long[df_long['Time'] > 15]

    return df_long


def merge_group_data(suffix):
    """
    把该组的 4 个指标合并在一起
    """
    print(f"正在读取 {suffix} 组数据...")

    # 读取 4 个 Sheet
    df_tcore = process_sheet_with_gender(excel_path, SHEET_MAP[f'Tcore{suffix}'], 'Tcore')
    df_vo2 = process_sheet_with_gender(excel_path, SHEET_MAP[f'VO2{suffix}'], 'VO2')
    df_act = process_sheet_with_gender(excel_path, SHEET_MAP[f'Activity{suffix}'], 'Activity')
    df_hr = process_sheet_with_gender(excel_path, SHEET_MAP[f'HR{suffix}'], 'Heart_Rate')

    if df_tcore is None:
        return None

    # 合并数据 (注意：现在要把 'Gender' 也作为合并的钥匙，确保一一对应)
    merged = df_tcore

    # 定义合并的列 (Keys)
    merge_keys = ['Time', 'MouseID', 'Gender']

    if df_vo2 is not None: merged = pd.merge(merged, df_vo2, on=merge_keys, how='inner')
    if df_act is not None: merged = pd.merge(merged, df_act, on=merge_keys, how='inner')
    if df_hr is not None: merged = pd.merge(merged, df_hr, on=merge_keys, how='inner')

    return merged


# ==========================================
# 3. 执行合并与构建数据集
# ==========================================
# 处理 Torpor 组
df_torpor = merge_group_data('_Torpor')
if df_torpor is not None:
    df_torpor['Label'] = 1

# 处理 Non-Torpor 组
df_nontorpor = merge_group_data('_NonTorpor')
if df_nontorpor is not None:
    df_nontorpor['Label'] = 0

# 拼接
if df_torpor is None and df_nontorpor is None:
    print("❌ 错误：两组数据都读取失败，请检查 Sheet 名字是否正确！")
else:
    final_df = pd.concat([df_torpor, df_nontorpor], ignore_index=True)
    final_df = final_df.dropna()

    print(f"\n✅ 数据准备完成！共 {len(final_df)} 个有效样本点。")
    print(final_df.head())

    # ==========================================
    # 4. 运行 LASSO 模型
    # ==========================================
    # 加入 Gender
    features = ['Tcore', 'VO2', 'Activity', 'Heart_Rate', 'Gender']

    X = final_df[features].values
    y = final_df['Label'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练 LASSO
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1, random_state=42)
    lasso.fit(X_scaled, y)

    # ==========================================
    # 5. 结果可视化 (数字居中)
    # ==========================================
    coefs = lasso.coef_[0]

    plt.figure(figsize=(10, 6))
    colors = ['blue' if c < 0 else 'red' for c in coefs]
    bars = plt.bar(features, coefs, color=colors)

    plt.title('LASSO Feature Selection (with Gender)', fontsize=14)
    plt.ylabel('LASSO Coefficient (Importance)', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, v in zip(bars, coefs):
        y_pos = v / 2
        text_color = 'white' if abs(v) > 0.2 else 'black'

        if abs(v) > 0.001:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     y_pos,
                     f"{v:.2f}",
                     ha='center',
                     va='center',
                     color=text_color,
                     fontweight='bold',
                     fontsize=12)

    plt.tight_layout()
    plt.show()