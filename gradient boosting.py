# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.inspection import permutation_importance
# import os
#
# # ==========================================
# # 1. 配置：Excel 文件名与 Sheet 名称映射
# # ==========================================
# # 请确保这个文件名和你电脑里的 Excel 文件名完全一致
# EXCEL_FILENAME = "Organized data.xlsx"
#
# # 这里的 Key 是 Sheet 的名字，Value 是 (特征类型, 分组)
# # 请务必打开你的 Excel，检查底部的 Tab 名字是否和下面引号里的一模一样！
# # 注意大小写和空格（比如 "ativity" 的拼写错误）
# sheet_config = {
#     "Tcore torpor-like": ("Tcore", "Torpor"),
#     "Tcore nontorpor": ("Tcore", "Non-torpor"),
#     "VO2 torpor-like": ("VO2", "Torpor"),
#     "VO2 nontorpor": ("VO2", "Non-torpor"),
#     "activity torpor-like": ("Activity", "Torpor"),
#     "ativity nontorpor": ("Activity", "Non-torpor"),  # 注意这里保留了你之前文件里的拼写 "ativity"
#     "Heart Rate torpor-like": ("HeartRate", "Torpor"),
#     "Heart Rate nontorpor": ("HeartRate", "Non-torpor")
# }
#
#
# # ==========================================
# # 2. 数据处理函数
# # ==========================================
# def load_and_process_excel(filename, config):
#     print(f"正在读取 Excel 文件: {filename} ...")
#
#     if not os.path.exists(filename):
#         raise FileNotFoundError(f"❌ 找不到文件 '{filename}'。请确认文件放在了 {os.getcwd()} 目录下。")
#
#     all_dfs = []
#
#     # 读取 Excel 文件（需要安装 openpyxl: pip install openpyxl）
#     try:
#         xls = pd.ExcelFile(filename)
#     except Exception as e:
#         raise ValueError(f"❌ 无法打开 Excel 文件。请确保文件未被其他程序占用，且已安装 openpyxl。\n错误信息: {e}")
#
#     # 打印所有可用的 Sheet 名字，方便调试
#     print(f"Excel 中包含的 Sheet: {xls.sheet_names}")
#
#     for sheet_name, (feat_name, group_label) in config.items():
#         if sheet_name not in xls.sheet_names:
#             print(f"⚠️ 警告: 找不到名为 '{sheet_name}' 的 Sheet。跳过此项。")
#             continue
#
#         try:
#             # 读取指定的 Sheet
#             df_raw = pd.read_excel(xls, sheet_name=sheet_name)
#
#             # --- 数据清洗逻辑 (同之前) ---
#             # 1. 检查并移除性别行 (第一行如果是 M/F)
#             first_row = df_raw.iloc[0].astype(str).values
#             has_sex_row = np.any([x in ['M', 'F'] for x in first_row])
#
#             sex_map = {}
#             if has_sex_row:
#                 for col in df_raw.columns:
#                     val = df_raw.iloc[0][col]
#                     if val in ['M', 'F']:
#                         clean_col = str(col).replace('Average ', '')
#                         sex_map[clean_col] = val
#                 df_data = df_raw.drop(0).copy()
#             else:
#                 df_data = df_raw.copy()
#
#             # 2. 清理列名
#             df_data.columns = [str(c).replace('Average ', '') for c in df_data.columns]
#
#             # 3. 找时间列
#             time_cols = [c for c in df_data.columns if 'time' in str(c).lower()]
#             if not time_cols:
#                 # 找不到时间列就用第一列
#                 time_col = df_data.columns[0]
#             else:
#                 time_col = time_cols[0]
#
#             df_data = df_data.rename(columns={time_col: 'Time'})
#             df_data['Time'] = pd.to_numeric(df_data['Time'], errors='coerce')
#
#             # 4. 宽表转长表
#             subj_cols = [c for c in df_data.columns if c != 'Time' and 'Unnamed' not in str(c)]
#             df_long = df_data.melt(id_vars=['Time'], value_vars=subj_cols,
#                                    var_name='Subject_ID', value_name=feat_name)
#
#             # 5. 添加元数据
#             df_long['Group'] = group_label
#             df_long['Sex'] = df_long['Subject_ID'].map(sex_map)
#             df_long[feat_name] = pd.to_numeric(df_long[feat_name], errors='coerce')
#
#             all_dfs.append(df_long)
#             print(f"✅ 成功处理 Sheet: {sheet_name}")
#
#         except Exception as e:
#             print(f"❌ 处理 Sheet '{sheet_name}' 时出错: {e}")
#
#     if not all_dfs:
#         raise ValueError("没有成功处理任何数据！请检查 Sheet 名称是否正确。")
#
#     # 合并
#     merged = pd.concat(all_dfs, ignore_index=True)
#     final = merged.groupby(['Time', 'Group', 'Subject_ID'])[
#         ['Tcore', 'VO2', 'Activity', 'HeartRate', 'Sex']].first().reset_index()
#     return final
#
#
# # ==========================================
# # 3. 主程序执行
# # ==========================================
#
# # 1. 加载数据
# try:
#     df = load_and_process_excel(EXCEL_FILENAME, sheet_config)
# except Exception as e:
#     print(e)
#     exit()
#
# # 2. 准备训练数据 (Non-torpor)
# train_df = df[df['Group'] == 'Non-torpor'].copy()
#
# # 简单的性别编码
# train_df['Sex_Encoded'] = train_df['Sex'].map({'M': 0, 'F': 1}).fillna(0)
#
# features = ['Time', 'Tcore', 'Activity', 'HeartRate', 'Sex_Encoded']
# target = 'VO2'
#
# # 移除目标值为空的行
# train_clean = train_df.dropna(subset=[target])
# X = train_clean[features]
# y = train_clean[target]
#
# print(f"\n训练集样本数: {len(X)}")
#
# if len(X) == 0:
#     print("❌ 错误：训练集为空。可能是因为 Excel 数据读取有问题，或者 Group 分组名称不对。")
#     exit()
#
# # 3. 训练模型
# print("正在训练模型...")
# model = HistGradientBoostingRegressor(random_state=42, max_iter=200, learning_rate=0.1)
# model.fit(X, y)
#
# # 4. 计算特征重要性
# r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
#
# # 5. 可视化
# sorted_idx = r.importances_mean.argsort()
#
# plt.figure(figsize=(10, 6))
# # 设置中文字体（防止乱码，根据你的系统选择）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.barh([features[i] for i in sorted_idx], r.importances_mean[sorted_idx],
#          xerr=r.importances_std[sorted_idx], color='skyblue', edgecolor='black')
#
# plt.title('Non-torpor VO2 预测模型 - 特征重要性')
# plt.xlabel('重要性得分 (Importance)')
# plt.tight_layout()
# plt.show()
#
# print("\n分析完成！")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import mean_squared_error

# =================================================================
# 1. DATA LOADING CONFIGURATION
# =================================================================
# Ensure your Excel file is named "Organized data.xlsx" in the same folder
EXCEL_FILENAME = "Organized data.xlsx"

# Mapping Sheets to Features and Groups
# Verify these sheet names match your Excel tabs exactly!
sheet_config = {
    "Tcore torpor-like": ("Tcore", "Torpor"),
    "Tcore nontorpor": ("Tcore", "Non-torpor"),
    "VO2 torpor-like": ("VO2", "Torpor"),
    "VO2 nontorpor": ("VO2", "Non-torpor"),
    "activity torpor-like": ("Activity", "Torpor"),
    "ativity nontorpor": ("Activity", "Non-torpor"),
    "Heart Rate torpor-like": ("HeartRate", "Torpor"),
    "Heart Rate nontorpor": ("HeartRate", "Non-torpor")
}


def load_and_process_data(filename, config):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found in {os.getcwd()}")

    xls = pd.ExcelFile(filename)
    all_dfs = []

    for sheet_name, (feat_name, group_label) in config.items():
        if sheet_name not in xls.sheet_names:
            print(f"Warning: Sheet '{sheet_name}' missing. Skipping...")
            continue

        # Read sheet
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)

        # Metadata extraction (Sex row)
        first_row = df_raw.iloc[0].astype(str).values
        has_sex_row = np.any([x in ['M', 'F'] for x in first_row])

        sex_map = {}
        if has_sex_row:
            for col in df_raw.columns:
                val = df_raw.iloc[0][col]
                if val in ['M', 'F']:
                    clean_col = str(col).replace('Average ', '')
                    sex_map[clean_col] = val
            df_data = df_raw.drop(0).copy()
        else:
            df_data = df_raw.copy()

        # Column cleaning
        df_data.columns = [str(c).replace('Average ', '') for c in df_data.columns]
        time_cols = [c for c in df_data.columns if 'time' in str(c).lower()]
        time_col = time_cols[0] if time_cols else df_data.columns[0]
        df_data = df_data.rename(columns={time_col: 'Time'})

        # Melt to long format
        subj_cols = [c for c in df_data.columns if c != 'Time' and 'Unnamed' not in str(c)]
        df_long = df_data.melt(id_vars=['Time'], value_vars=subj_cols,
                               var_name='Subject_ID', value_name=feat_name)

        df_long['Group'] = group_label
        df_long['Sex'] = df_long['Subject_ID'].map(sex_map)
        df_long[feat_name] = pd.to_numeric(df_long[feat_name], errors='coerce')
        all_dfs.append(df_long)

    merged = pd.concat(all_dfs, ignore_index=True)
    final = merged.groupby(['Time', 'Group', 'Subject_ID'])[
        ['Tcore', 'VO2', 'Activity', 'HeartRate', 'Sex']].first().reset_index()
    return final


# =================================================================
# 2. FEATURE ENGINEERING & PREPARATION
# =================================================================
try:
    df = load_and_process_data(EXCEL_FILENAME, sheet_config)
    # Filter for Non-torpor (The Baseline Model)
    train_df = df[df['Group'] == 'Non-torpor'].copy()
    train_df['Sex_Encoded'] = train_df['Sex'].map({'M': 0, 'F': 1}).fillna(0)

    # CREATE INTERACTION FEATURES (Combining Features)
    train_df['Tcore_x_HeartRate'] = train_df['Tcore'] * train_df['HeartRate']
    train_df['Tcore_x_Activity'] = train_df['Tcore'] * train_df['Activity']
    train_df['HeartRate_x_Activity'] = train_df['HeartRate'] * train_df['Activity']
    train_df['Tcore_x_Time'] = train_df['Tcore'] * train_df['Time']

    features = [
        'Time', 'Tcore', 'Activity', 'HeartRate', 'Sex_Encoded',
        'Tcore_x_HeartRate', 'Tcore_x_Activity', 'HeartRate_x_Activity', 'Tcore_x_Time'
    ]
    target = 'VO2'

    train_clean = train_df.dropna(subset=[target]).copy()
    X = train_clean[features]
    y = train_clean[target]
    groups = train_clean['Subject_ID']

except Exception as e:
    print(f"Error: {e}")
    exit()

# =================================================================
# 3. LEAVE-ONE-SUBJECT-OUT CROSS VALIDATION (RMSE)
# =================================================================
print("\n--- Starting Leave-One-Subject-Out Cross-Validation ---")
model = HistGradientBoostingRegressor(random_state=42)
logo = LeaveOneGroupOut()

# Calculate Cross-Validated RMSE
# Scoring is negative RMSE in sklearn, so we flip the sign
cv_scores = cross_val_score(model, X, y, groups=groups, cv=logo,
                            scoring='neg_root_mean_squared_error')

rmse_scores = -cv_scores
print(f"Mean RMSE: {rmse_scores.mean():.4f}")
print(f"RMSE Std:  {rmse_scores.std():.4f}")

# =================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# =================================================================
print("\n--- Calculating Feature Importance (Permutation) ---")
model.fit(X, y)
result = permutation_importance(model, X, y, n_repeats=10, random_state=42)

# Prepare DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': result.importances_mean,
    'Std': result.importances_std
}).sort_values(by='Importance', ascending=True)

# =================================================================
# 5. VISUALIZATION
# =================================================================
plt.figure(figsize=(12, 8))
# Differentiate colors for original vs interaction features
colors = ['#f39c12' if 'x' in feat else '#3498db' for feat in importance_df['Feature']]

plt.barh(importance_df['Feature'], importance_df['Importance'],
         xerr=importance_df['Std'], color=colors, edgecolor='black', alpha=0.8)

plt.title('VO2 Prediction Model: Feature Importance (Including Interactions)', fontsize=15)
plt.xlabel('Importance Score (Drop in Model Performance)', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Custom Legend
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor='#3498db', label='Original Features'),
                   Patch(facecolor='#f39c12', label='Interaction Terms (Combination)')]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()

print("\n--- Final Report ---")
print(f"Evaluation Metric (RMSE): {rmse_scores.mean():.4f}")
print("Top 5 Contributing Factors:")
print(importance_df.sort_values(by='Importance', ascending=False).head(5))