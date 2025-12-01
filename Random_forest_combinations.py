import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

train_files = [
    "M377_merged.xlsx",
    "M379_merged.xlsx",
    "M380_merged.xlsx"
]

test_files = [
    "M382_merged.xlsx",
    "M383_merged.xlsx"
]

def load_mouse_data(path):
    df = pd.read_excel(path)
    if "Sleep" in df.columns:
        df = df[df["Sleep"] != "X"]
    numeric_cols = [
        "TEE", "Tb_dsi", "Activity_dsi", "Food_intake",
        "Ta(ambient_temp)", "age", "BW_start", "BW_end"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "BW_end" in df.columns:
        df["BW_mean"] = (df["BW_start"] + df["BW_end"]) / 2.0
    else:
        df["BW_mean"] = df["BW_start"]
    df = df.dropna(subset=["TEE", "Tb_dsi", "Activity_dsi"])
    return df

def add_interaction_features(df):
    df = df.copy()
    df["Tb_x_Ta"] = df["Tb_dsi"] * df["Ta(ambient_temp)"]
    df["Sleep_W"] = (df["Sleep"] == "W").astype(int)
    df["Tb_x_Sleep"] = df["Tb_dsi"] * df["Sleep_W"]
    df["Phase_dark"] = (df["Phase_lbl"] == "dark").astype(int)
    df["Phase_x_Sleep"] = df["Phase_dark"] * df["Sleep_W"]
    df["Activity_x_Tb"] = df["Activity_dsi"] * df["Tb_dsi"]
    df["Food_x_Phase"] = df["Food_intake"] * df["Phase_dark"]
    return df

train_dfs = [load_mouse_data(f) for f in train_files]
df_train = pd.concat(train_dfs, ignore_index=True)

test_dfs = [load_mouse_data(f) for f in test_files]
df_test = pd.concat(test_dfs, ignore_index=True)

df_train = add_interaction_features(df_train)
df_test = add_interaction_features(df_test)

numeric_features = [
    "Tb_dsi",
    "Activity_dsi",
    "Food_intake",
    "Ta(ambient_temp)",
    "age",
    "BW_mean",
    "Tb_x_Ta",
    "Tb_x_Sleep",
    "Phase_x_Sleep",
    "Activity_x_Tb",
    "Food_x_Phase"
]

categorical_features = ["Phase_lbl", "Sleep"]
target = "TEE"

X_train_raw = df_train[numeric_features + categorical_features]
y_train = df_train[target]

X_test_raw = df_test[numeric_features + categorical_features]
y_test = df_test[target]

X_train = pd.get_dummies(X_train_raw, columns=categorical_features, drop_first=False)
X_test = pd.get_dummies(X_test_raw, columns=categorical_features, drop_first=False)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)
rf = grid.best_estimator_

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Tuned RF Cross-Mouse Validation ===")
print("MAE:", mae)
print("R² :", r2)
print("\nBest parameters:", grid.best_params_)

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

print("\n=== Feature Importances ===")
print(importance_df.head(20))

sorted_idx = r.importances_mean.argsort()

plt.figure(figsize=(12, 7))
plt.boxplot(
    r.importances[sorted_idx].T,
    vert=False,
    tick_labels=X_train.columns[sorted_idx]
)
plt.title("Permutation Importances (Cross-Mouse Test Set)")
plt.tight_layout()
plt.savefig("permutation_importance_combinations_tuned.png", dpi=300)


