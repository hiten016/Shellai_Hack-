import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
feature_cols = train.columns.difference(target_cols).tolist()

def engineer_features(df):
    df = df.copy()
    comp_cols = [f"Component{i}_fraction" for i in range(1, 6)]
    df["comp_mean"] = df[comp_cols].mean(axis=1)
    df["comp_std"] = df[comp_cols].std(axis=1)
    for i in range(1, 6):
        props = [f"Component{i}_Property{j}" for j in range(1, 11)]
        df[f"Component{i}_mean"] = df[props].mean(axis=1)
        df[f"Component{i}_std"] = df[props].std(axis=1)
    for j in range(1, 11):
        weighted = 0
        for i in range(1, 6):
            vol = df[f"Component{i}_fraction"]
            prop = df[f"Component{i}_Property{j}"]
            weighted += vol * prop
        df[f"weighted_prop_{j}"] = weighted
    return df

train = engineer_features(train)
test = engineer_features(test)

X = train.drop(columns=target_cols)
y = train[target_cols]
X_test = test[X.columns]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_preds = pd.DataFrame({"ID": test["ID"]})
lgb_preds = pd.DataFrame({"ID": test["ID"]})
ensemble_preds = pd.DataFrame({"ID": test["ID"]})

total_mape_xgb = 0
total_mape_lgb = 0

for target in target_cols:
    print(f"\n🔍 Tuning XGBoost for {target}...")

    def objective_xgb(trial, X, y_target):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "n_jobs": -1
        }
        model = XGBRegressor(**params)
        mape = -1.0 * cross_val_score(model, X, y_target, scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False), cv=3).mean()
        return mape

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y[target]), n_trials=30, timeout=600)
    best_params_xgb = study_xgb.best_params
    print(f" Best XGBoost params for {target}: {best_params_xgb}")

    oof_xgb = np.zeros(len(train))
    test_preds_xgb = np.zeros(len(test))
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]
        model = XGBRegressor(**best_params_xgb)
        model.fit(X_train, y_train)
        oof_xgb[val_idx] = model.predict(X_val)
        test_preds_xgb += model.predict(X_test) / kf.n_splits
    mape_xgb = mean_absolute_percentage_error(y[target], oof_xgb)
    total_mape_xgb += mape_xgb
    print(f" {target} XGBoost MAPE: {mape_xgb:.4f}")
    xgb_preds[target] = test_preds_xgb

    print(f"\n Tuning LightGBM for {target}...")

    def objective_lgb(trial, X, y_target):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "random_state": 42,
            "n_jobs": -1
        }
        model = LGBMRegressor(**params)
        mape = -1.0 * cross_val_score(model, X, y_target, scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False), cv=3).mean()
        return mape

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, X, y[target]), n_trials=30, timeout=600)
    best_params_lgb = study_lgb.best_params
    print(f" Best LightGBM params for {target}: {best_params_lgb}")

    oof_lgb = np.zeros(len(train))
    test_preds_lgb = np.zeros(len(test))
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]
        model = LGBMRegressor(**best_params_lgb)
        model.fit(X_train, y_train)
        oof_lgb[val_idx] = model.predict(X_val)
        test_preds_lgb += model.predict(X_test) / kf.n_splits
    mape_lgb = mean_absolute_percentage_error(y[target], oof_lgb)
    total_mape_lgb += mape_lgb
    print(f" {target} LightGBM MAPE: {mape_lgb:.4f}")
    lgb_preds[target] = test_preds_lgb

    ensemble_preds[target] = 0.5 * test_preds_xgb + 0.5 * test_preds_lgb

xgb_preds.to_csv("xgb_optuna_submission.csv", index=False)
lgb_preds.to_csv("lgb_optuna_submission.csv", index=False)
ensemble_preds.to_csv("xgb_lgb_ensemble_submission.csv", index=False)

print(f"\n Avg XGBoost MAPE: {total_mape_xgb / len(target_cols):.4f}")
print(f" Avg LightGBM MAPE: {total_mape_lgb / len(target_cols):.4f}")
