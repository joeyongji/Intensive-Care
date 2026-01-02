# Predict VO2 (oxygen consumption) from Time, Tcore, Activity, HeartRate, Sex
# - Trains on Non-torpor only
# - Tunes RF ONCE using GroupKFold (no leakage across subjects)
# - Evaluates with Leave-One-Subject-Out (LOSO)
# - Computes permutation importance on held-out subjects
# - Stabilizes importance by averaging across multiple random seeds


import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance


XLSX_PATH = "Organized data.xlsx"

sheet_config = {
    "Tcore torpor-like": ("Tcore", "Torpor"),
    "Tcore nontorpor": ("Tcore", "Non-torpor"),
    "VO2 torpor-like": ("VO2", "Torpor"),
    "VO2 nontorpor": ("VO2", "Non-torpor"),
    "activity torpor-like": ("Activity", "Torpor"),
    "ativity nontorpor": ("Activity", "Non-torpor"),
    "Heart Rate torpor-like": ("HeartRate", "Torpor"),
    "Heart Rate nontorpor": ("HeartRate", "Non-torpor"),
}


RANDOM_STATE = 0
N_ITER_SEARCH = 12


SEEDS = [0, 1, 2, 3, 4]   # multiple seeds
N_REPEATS_PERM = 50       # permutation repeats per seed

# Inner CV for tuning
INNER_SPLITS = 5


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# sklearn expects higher is better -> use negative RMSE
neg_rmse_scorer = make_scorer(lambda yt, yp: -rmse(yt, yp), greater_is_better=True)


def load_and_process_data(xlsx_path: str, config: dict) -> pd.DataFrame:
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"File not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    all_dfs = []

    for sheet_name, (feat_name, group_label) in config.items():
        if sheet_name not in xls.sheet_names:
            print(f" Missing sheet '{sheet_name}'.")
            continue

        df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        # Row 0 = mouse IDs, row 1 = sex labels, row 2+ = time series
        mouse_ids = df_raw.iloc[0, 1:].astype(str).tolist()
        sex_labels = df_raw.iloc[1, 1:].astype(str).tolist()
        sex_map = dict(zip(mouse_ids, sex_labels))

        time = pd.to_numeric(df_raw.iloc[2:, 0], errors="coerce").astype(float)
        vals = df_raw.iloc[2:, 1:]
        vals.columns = mouse_ids
        vals = vals.apply(pd.to_numeric, errors="coerce")

        df_long = vals.copy()
        df_long.insert(0, "Time", time.values)
        df_long = df_long.melt(id_vars=["Time"], var_name="Subject_ID", value_name=feat_name)

        df_long["Group"] = group_label
        df_long["Sex"] = df_long["Subject_ID"].map(sex_map)

        all_dfs.append(df_long)

    merged = pd.concat(all_dfs, ignore_index=True)

    # One row per timepoint per subject per group, with all features merged
    final = merged.groupby(["Time", "Group", "Subject_ID"], as_index=False).first()
    return final

# Tuning
def tune_once(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> RandomForestRegressor:
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    # Small param space for speed
    param_dist = {
        "n_estimators": [300, 600, 900],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.7, 1.0],
    }

    n_groups = len(np.unique(groups))
    n_splits = min(INNER_SPLITS, n_groups) if n_groups >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=neg_rmse_scorer,
        cv=gkf.split(X, y, groups=groups),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X, y)
    print("Best params:", search.best_params_)
    return search.best_estimator_


def loso_importance_multiseed(
    tuned_model: RandomForestRegressor,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    label: str
) -> pd.DataFrame:
    logo = LeaveOneGroupOut()

    rmses = []
    perm_means_all = []  # each element is (n_features,) for (fold, seed)

    # We re-fit per seed for stability (more principled). Still pretty fast with 11 subjects.
    base_params = tuned_model.get_params()

    for tr, te in logo.split(X, y, groups=groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        for seed in SEEDS:
            m = RandomForestRegressor(**base_params)
            m.set_params(random_state=seed)
            m.fit(Xtr, ytr)

            pred = m.predict(Xte)
            rmses.append(rmse(yte, pred))

            r = permutation_importance(
                m,
                Xte, yte,
                n_repeats=N_REPEATS_PERM,
                random_state=seed,
                scoring=neg_rmse_scorer
            )
            perm_means_all.append(r.importances_mean)

    rmses = np.array(rmses, dtype=float)
    print(f"\n[{label}] LOSO RMSE (folds×seeds): mean={rmses.mean():.4f}, std={rmses.std():.4f}")

    perm_mat = np.vstack(perm_means_all)  # (n_runs, n_features)

    mean_imp = perm_mat.mean(axis=0)
    std_imp = perm_mat.std(axis=0)
    ci_low = np.quantile(perm_mat, 0.025, axis=0)
    ci_high = np.quantile(perm_mat, 0.975, axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance_mean": mean_imp,
        "Std": std_imp,
        "CI_low": ci_low,
        "CI_high": ci_high
    }).sort_values("Importance_mean", ascending=False)

    print(f"\n[{label}] Feature Importances (LOSO permutation, averaged across seeds):")
    print(importance_df)

    csv_name = f"rf_vo2_importance_{label}_multi_seed.csv"
    png_name = f"rf_vo2_importance_{label}_multi_seed.png"

    importance_df.to_csv(csv_name, index=False)


    plot_df = importance_df.sort_values("Importance_mean", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Feature"], plot_df["Importance_mean"])
    plt.title(f"RF VO2 permutation importance (LOSO, multi-seed) — {label}")
    plt.xlabel("Importance (drop in -RMSE when permuted)")
    plt.tight_layout()
    plt.savefig(png_name, dpi=300)
    plt.close()

    print(f"\nSaved: {csv_name}")
    print(f"Saved: {png_name}")

    return importance_df



# MAIN

if __name__ == "__main__":
    df = load_and_process_data(XLSX_PATH, sheet_config)

    # Like your teammate: fit baseline model on Non-torpor only
    train_df = df[df["Group"] == "Non-torpor"].copy()
    # If you want ALL data:
    # train_df = df.copy()

    # Encode sex
    train_df["Sex_Encoded"] = train_df["Sex"].map({"M": 0, "F": 1}).fillna(0).astype(int)

    target = "VO2"

    # Ensure numeric columns
    for c in ["Time", "Tcore", "Activity", "HeartRate", "Sex_Encoded", target]:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

    # Drop missing
    train_clean = train_df.dropna(subset=["Time", "Tcore", "Activity", "HeartRate", "Sex_Encoded", target]).copy()

    print("Rows:", len(train_clean), "Subjects:", train_clean["Subject_ID"].nunique())

    y = train_clean[target].values
    groups = train_clean["Subject_ID"].values

    # WITH TIME
    features_with_time = ["Time", "Tcore", "Activity", "HeartRate", "Sex_Encoded"]
    X1 = train_clean[features_with_time].copy()

    tuned1 = tune_once(X1, y, groups)

    loso_importance_multiseed(
        tuned_model=tuned1,
        X=X1, y=y, groups=groups,
        feature_names=features_with_time,
        label="with_time"
    )

    # WITHOUT TIME
    features_no_time = ["Tcore", "Activity", "HeartRate", "Sex_Encoded"]
    X2 = train_clean[features_no_time].copy()

    tuned2 = tune_once(X2, y, groups)

    loso_importance_multiseed(
        tuned_model=tuned2,
        X=X2, y=y, groups=groups,
        feature_names=features_no_time,
        label="no_time"
    )
