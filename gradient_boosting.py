import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# 1. Configuration (Please modify with your actual filenames)
# ============================================
# Assuming the first three are the training set, and the last two are the test set
train_files = [
    "M377_merged.csv",
    "M380_merged.csv",
    "M379_merged.csv",
     # Example filenames, please modify
]

test_files = [
    "M382_merged.csv",  # Example filename, please modify
    "M383_merged.csv"   # Example filename, please modify
]


# ============================================
# 2. Define Data Preprocessing Function
# ============================================
def process_mouse_data(file_path):
    try:
        # Read data
        df = pd.read_csv(file_path, parse_dates=['date_time_corrected'])
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found, skipping.")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 2.1 Remove Sleep='X'
    df = df[df['Sleep'] != 'X'].copy()

    # 2.2 Convert to numeric
    cols_to_numeric = ['VO2', 'Tb_dsi', 'Activity_dsi', 'Food_intake',
                       'Ta(ambient_temp)', 'age', 'BW_start', 'BW_end', 'Wake']
    for col in cols_to_numeric:
        # Error handling: If column missing (some files might differ), skip or handle
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2.3 Normalize VO2
    if 'BW_start' in df.columns and 'BW_end' in df.columns:
        df["BW_mean"] = (df["BW_start"] + df["BW_end"]) / 2.0
        df["VO2_norm"] = df["VO2"] / df["BW_mean"]

    # 2.4 Drop missing values
    df_model = df.dropna(subset=["VO2_norm", "Tb_dsi", "Activity_dsi", "Wake"]).copy()

    return df_model


# ============================================
# 3. Build Training and Testing Sets
# ============================================
print("Processing training data...")
train_dfs = []
for f in train_files:
    processed_df = process_mouse_data(f)
    if processed_df is not None:
        train_dfs.append(processed_df)

if not train_dfs:
    print("Error: No training data loaded!")
    exit()

df_train_all = pd.concat(train_dfs, ignore_index=True)
print(f"Total training rows: {len(df_train_all)}")

print("Processing testing data...")
test_dfs = []
for f in test_files:
    processed_df = process_mouse_data(f)
    if processed_df is not None:
        test_dfs.append(processed_df)

if not test_dfs:
    print("Error: No testing data loaded!")
    exit()

df_test_all = pd.concat(test_dfs, ignore_index=True)
print(f"Total testing rows: {len(df_test_all)}")

# ============================================
# 4. Feature Engineering
# ============================================
numeric_features = ["Tb_dsi", "Activity_dsi", "Food_intake", "Ta(ambient_temp)", "age", "Wake"]
categorical_features = ["Phase_lbl", "Sleep"]

# Prepare training data X, y
X_train_raw = df_train_all[numeric_features + categorical_features]
y_train = df_train_all["VO2_norm"]
X_train = pd.get_dummies(X_train_raw, columns=categorical_features, drop_first=False)

# Prepare testing data X, y
X_test_raw = df_test_all[numeric_features + categorical_features]
y_test = df_test_all["VO2_norm"]
X_test = pd.get_dummies(X_test_raw, columns=categorical_features, drop_first=False)

# Ensure columns in training and testing sets are identical (prevent missing state errors)
# This is crucial; e.g., if train set has REM sleep but test set doesn't, columns mismatch
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# ============================================
# 5. Train Model
# ============================================
print("Starting model training...")
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)

gbr.fit(X_train, y_train)

# ============================================
# 6. Evaluation and Visualization
# ============================================
y_pred = gbr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Cross-Subject Validation Results (Train: 3 mice, Test: 2 mice) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")

# Visualization: Plot a segment of the test set (combining two mice makes full plot messy)
plt.figure(figsize=(15, 6))
# Take the first 1000 points (roughly a segment of the first test mouse)
limit = 1000
plt.plot(np.arange(limit), y_test.values[:limit], label='Actual VO2', color='black', alpha=0.6)
plt.plot(np.arange(limit), y_pred[:limit], label='Predicted VO2', color='red', linestyle='--', alpha=0.8)
plt.title('Cross-Subject Validation: Actual vs Predicted VO2 (First 1000 points of Test Set)')
plt.xlabel('Time Points')
plt.ylabel('VO2 / BW')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": gbr.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Key Influencing Factors ===")
print(feature_importance.head(10)) 