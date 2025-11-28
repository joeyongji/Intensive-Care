import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

# --- Data files ---
train_files = ["M377_merged.xlsx", "M379_merged.xlsx", "M380_merged.xlsx"]
test_files  = ["M382_merged.xlsx", "M383_merged.xlsx"]

def load_mouse(file_path):
    df = pd.read_excel(file_path)
    # Clean
    df = df[df["Sleep"] != "X"] if "Sleep" in df.columns else df
    numeric = ["VO2", "Tb_dsi", "Activity_dsi", "Food_intake",
               "Ta(ambient_temp)", "age", "BW_start", "BW_end"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # body weight combined
    df["BW_mean"] = (df["BW_start"] + df["BW_end"]) / 2
    df = df.dropna(subset=["VO2", "Tb_dsi", "Activity_dsi"])
    return df

# Load
df_train = pd.concat([load_mouse(f) for f in train_files], ignore_index=True)
df_test  = pd.concat([load_mouse(f) for f in test_files], ignore_index=True)

features = [
    "Tb_dsi",
    "Activity_dsi",
    "Food_intake",
    "Ta(ambient_temp)",
    "Wake",
    "BW_mean"
]
target = "VO2"  # Changed from TEE to VO2

X_train = df_train[features]
y_train = df_train[target]
X_test  = df_test[features]
y_test  = df_test[target]

# --- Model ---
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))

# --- Permutation Importance ---
r = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=0
)

# --- Publication-Quality Plot ---
# Sort features by importance
sorted_idx = r.importances_mean.argsort()

# Set publication-quality style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Get sorted values
means = r.importances_mean[sorted_idx]
stds = r.importances_std[sorted_idx]

# Create color gradient (red = low importance, blue = high importance)
colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(means)))

# Create horizontal bars with error bars
y_pos = np.arange(len(means))
ax.barh(y_pos, means, xerr=stds,
        color=colors, edgecolor='#2c3e50', linewidth=1.2,
        error_kw={'linewidth': 2, 'ecolor': '#34495e', 'capsize': 4})

# Set feature names on y-axis
ax.set_yticks(y_pos)
ax.set_yticklabels(np.array(features)[sorted_idx], fontsize=12)

# Labels and title
ax.set_xlabel('Mean Decrease in Model Performance (R²) ± Std Dev', fontsize=13, fontweight='bold')
ax.set_title('Permutation Feature Importance for VO2 Prediction (Test Set)',
             fontsize=14, fontweight='bold', pad=15)

# Add vertical line at zero
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add grid for readability
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Clean up spines (remove top and right borders)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save as high-resolution PNG
plt.savefig("rf_permutation_importance.png", dpi=300, bbox_inches='tight')

# Display
plt.show()
