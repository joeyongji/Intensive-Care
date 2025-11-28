import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. Set File Paths
# ==========================================
base_dir = '/Users/ivy/Desktop/MDM intensive care/'

file_list = [
    "M377_merged.xlsx",
    "M379_merged.xlsx",
    "M380_merged.xlsx",
    "M382_merged.xlsx",
    "M383_merged.xlsx"
]

# ==========================================
# 2. Define Variables (Updated)
# ==========================================
target_col = 'VO2'

# Updated feature list:
# 1. Removed Sleep, added Wake
# 2. Added BW_start and BW_end
feature_candidates = [
    'Tb_dsi',              # Body Temperature
    'Activity_dsi',        # Activity
    'Ta(ambient_temp)',    # Ambient Temperature
    'Food_intake',         # Food Intake
    'Wake',                # Wake Status (1=Wake, 0=Sleep)
    'BW_start',            # Start Body Weight
    'BW_end'               # End Body Weight
]

results = {}

# ==========================================
# 3. Loop Through Each Mouse
# ==========================================
print(f"Starting analysis for 5 mice (Features include: Wake, BW)...\n")

for filename in file_list:
    mouse_id = filename.split('_')[0]
    file_path = os.path.join(base_dir, filename)

    print(f"[{mouse_id}] Processing...")

    if not os.path.exists(file_path):
        print(f"   ❌ File not found: {filename}")
        continue

    try:
        # Read data
        df = pd.read_excel(file_path)

        # --- Data Preparation ---
        # 1. Filter existing columns
        available_features = [col for col in feature_candidates if col in df.columns]

        # 2. Extract data and drop missing values
        cols_to_use = [target_col] + available_features
        df_clean = df[cols_to_use].dropna()

        if len(df_clean) < 10:
            print("   ⚠️ Not enough data, skipping.")
            continue

        # 3. [Critical Step] Check and remove "constant columns" (Variance = 0)
        # Explanation: For a single mouse, BW_start might be constant (e.g., 25g) throughout,
        # which is meaningless for predicting "minute-by-minute changes".
        # If not removed, StandardScaler will raise an error (division by zero).
        non_constant_features = []
        for col in available_features:
            if df_clean[col].nunique() > 1:  # If this column has at least two unique values
                non_constant_features.append(col)
            else:
                # Silently skip or print a hint if needed
                pass
                # print(f"      -> Removing constant variable: {col} (No variation for this mouse)")

        if not non_constant_features:
            print("   ⚠️ No valid variable features, skipping.")
            continue

        # --- Run LASSO ---
        X = df_clean[non_constant_features]
        y = df_clean[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)

        # Save coefficients
        coefs = pd.Series(lasso.coef_, index=non_constant_features)
        results[mouse_id] = coefs

        # --- Plotting ---
        plt.figure(figsize=(8, 4))
        colors = ['red' if abs(x) < 1e-4 else '#2ca02c' for x in coefs]

        # Sort coefficients for better visualization
        coefs.sort_values().plot(kind='barh', color=colors)

        plt.title(f'LASSO Selection: {mouse_id}', fontsize=12, fontweight='bold')
        plt.xlabel('Importance (Coefficient)')
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"   ❌ Error: {e}")

# ==========================================
# 4. Generate Summary Plot
# ==========================================
if results:
    results_df = pd.DataFrame(results)

    # Count how many times each variable was selected (coefficient != 0)
    selection_count = (results_df.abs() > 1e-4).sum(axis=1)

    print("\n" + "=" * 30)
    print("      Summary Statistics")
    print("=" * 30)
    print(results_df)