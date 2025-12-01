import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# train 377, 380 and 379

# Try Combinations stuff also

train_files = [
    "M377_merged.xlsx",
    "M379_merged.xlsx",
    "M380_merged.xlsx"
]

test_files = [
    "M382_merged.xlsx",
    "M383_merged.xlsx"
]
def load_mouse_data(file_path):
    df = pd.read_excel(file_path)

    # Remove Sleep = X (if present)
    if "Sleep" in df.columns:
        df = df[df["Sleep"] != "X"]

    # Convert numeric columns
    numeric_cols = [
        "TEE", "Tb_dsi", "Activity_dsi", "Food_intake",
        "Ta(ambient_temp)", "age", "BW_start", "BW_end"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace missing BW_end with BW_start if needed
    if "BW_end" in df.columns:
        df["BW_mean"] = (df["BW_start"] + df["BW_end"]) / 2
    else:
        df["BW_mean"] = df["BW_start"]

    # Drop missing
    needed = ["TEE", "Tb_dsi", "Activity_dsi"]
    df = df.dropna(subset=needed)

    return df

train_dfs = [load_mouse_data(f) for f in train_files]
df_train = pd.concat(train_dfs, ignore_index=True)

test_dfs = [load_mouse_data(f) for f in test_files]
df_test = pd.concat(test_dfs, ignore_index=True)

print("Train rows:", len(df_train))
print("Test rows:", len(df_test))

features_numeric = [
    "Tb_dsi",
    "Activity_dsi",
    "Food_intake",
    "Ta(ambient_temp)",
    "age",
    "BW_mean"
]

features_categorical = [
    "Phase_lbl",
    "Sleep"
]

target = "TEE"

X_train_raw = df_train[features_numeric + features_categorical]
y_train = df_train[target]
X_test_raw = df_test[features_numeric + features_categorical]
y_test = df_test[target]

X_train = pd.get_dummies(X_train_raw, columns=features_categorical, drop_first=False)
X_test = pd.get_dummies(X_test_raw, columns=features_categorical, drop_first=False)

# Align columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nCross-Mouse Validation (RF)")
print("MAE:", mae)
print("R2 :", r2)

r = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=0
)

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": r.importances_mean,
    "Std": r.importances_std
}).sort_values("Importance", ascending=False)

print("\nTop Feature Importances:")
print(importance_df.head(10))

sorted_idx = r.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.boxplot(
    r.importances[sorted_idx].T,
    vert=False,
    tick_labels=X_train.columns[sorted_idx]
)
plt.title("Permutation Importances (Cross-Mouse Test Set)")
plt.tight_layout()
plt.savefig("permutation_importance_RF_crossmouse.png", dpi=300)
