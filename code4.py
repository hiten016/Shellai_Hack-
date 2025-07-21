import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import optuna
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
feature_cols = train.columns.difference(target_cols).tolist()

# ⚙ Feature Engineering (kept same structure, cleaner logic)
def engineer_features(df):
    df = df.copy()
    comp_cols = [f"Component{i}_fraction" for i in range(1, 6)]
    df["comp_mean"] = df[comp_cols].mean(axis=1)
    df["comp_std"] = df[comp_cols].std(axis=1)
    for i in range(1, 6):
        props = [f"Component{i}_Property{j}" for j in range(1, 11)]
        df[f"Component{i}_mean"] = df[props].mean(axis=1)
        df[f"Component{i}_std"] = df[props].std(axis=1)
    for j in range(1, 10 + 1):
        df[f"weighted_prop_{j}"] = sum(
            df[f"Component{i}_fraction"] * df[f"Component{i}_Property{j}"] for i in range(1, 6)
        )
    return df


train = engineer_features(train)
test = engineer_features(test)

X = train.drop(columns=target_cols)
y = train[target_cols]
X_test = test[X.columns]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions
histgb_preds = pd.DataFrame({"ID": test["ID"]})
lgb_preds = pd.DataFrame({"ID": test["ID"]})
ridge_ensemble = pd.DataFrame({"ID": test["ID"]})

total_mape_hist = 0
total_mape_lgb = 0
total_mape_ensemble = 0

# Loop over each target
for target in target_cols:
    print(f"\n Tuning HistGradientBoosting for {target}...")

    def objective_hist(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 30),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "max_bins": trial.suggest_int("max_bins", 128, 255),
            "random_state": 42
        }
        model = HistGradientBoostingRegressor(**params)
        mape = -cross_val_score(model, X, y[target], scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False), cv=3).mean()
        return mape

    study_hist = optuna.create_study(direction="minimize")
    study_hist.optimize(objective_hist, n_trials=25, timeout=300)
    best_hist_params = study_hist.best_params

    print(f" Best HGB Params for {target}: {best_hist_params}")

    oof_hist = np.zeros(len(train))
    test_preds_hist = np.zeros(len(test))

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]
        model = HistGradientBoostingRegressor(**best_hist_params)
        model.fit(X_tr, y_tr)
        oof_hist[val_idx] = model.predict(X_val)
        test_preds_hist += model.predict(X_test) / kf.n_splits

    mape_hist = mean_absolute_percentage_error(y[target], oof_hist)
    print(f" {target} HGB MAPE: {mape_hist:.4f}")
    total_mape_hist += mape_hist
    histgb_preds[target] = test_preds_hist

    print(f"\n Tuning LightGBM for {target}...")

    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
            "n_jobs": -1
        }
        model = LGBMRegressor(**params)
        mape = -cross_val_score(model, X, y[target], scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False), cv=3).mean()
        return mape

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=25, timeout=300)
    best_lgb_params = study_lgb.best_params

    print(f" Best LGB Params for {target}: {best_lgb_params}")

    oof_lgb = np.zeros(len(train))
    test_preds_lgb = np.zeros(len(test))

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]
        model = LGBMRegressor(**best_lgb_params)
        model.fit(X_tr, y_tr)
        oof_lgb[val_idx] = model.predict(X_val)
        test_preds_lgb += model.predict(X_test) / kf.n_splits

    mape_lgb = mean_absolute_percentage_error(y[target], oof_lgb)
    print(f" {target} LGB MAPE: {mape_lgb:.4f}")
    total_mape_lgb += mape_lgb
    lgb_preds[target] = test_preds_lgb

    # Meta-model ensemble (Ridge)
    print(f" Ensembling for {target} using Ridge Regression...")
    ridge = Ridge()
    meta_X = np.vstack([oof_hist, oof_lgb]).T
    meta_test = np.vstack([test_preds_hist, test_preds_lgb]).T
    ridge.fit(meta_X, y[target])
    final_preds = ridge.predict(meta_test)
    ridge_ensemble[target] = final_preds

    mape_ridge = mean_absolute_percentage_error(y[target], ridge.predict(meta_X))
    print(f" Ridge Ensemble MAPE for {target}: {mape_ridge:.4f}")
    total_mape_ensemble += mape_ridge

#  Save Outputs
histgb_preds.to_csv("histgb_optuna_submission.csv", index=False)
lgb_preds.to_csv("lgb_optuna_submission3.csv", index=False)
ridge_ensemble.to_csv("ensemble_ridge_submission.csv", index=False)

print(f"\n Avg HistGradientBoosting MAPE: {total_mape_hist / len(target_cols):.4f}")
print(f" Avg LightGBM MAPE: {total_mape_lgb / len(target_cols):.4f}")
print(f" Avg Ridge Ensemble MAPE: {total_mape_ensemble / len(target_cols):.4f}")