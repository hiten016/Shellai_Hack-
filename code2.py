import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from xgboost import XGBRegressor
import optuna
import warnings
warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Define targets and features
target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
feature_cols = train.columns.difference(target_cols).tolist()

# Feature engineering
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

# Optuna objective function
def objective(trial, X, y_target):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
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

# Train per target
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = pd.DataFrame()
predictions["ID"] = test["ID"]
total_mape = 0

for target in target_cols:
    print(f"\n🔍 Tuning for {target}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y[target]), n_trials=30, timeout=600)

    print(f"Best params for {target}: {study.best_params}")

    # 5-Fold CV using best params
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]

        model = XGBRegressor(**study.best_params)
        model.fit(X_train, y_train)
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / kf.n_splits

    mape = mean_absolute_percentage_error(y[target], oof)
    total_mape += mape
    print(f"📊 {target} MAPE: {mape:.4f}")
    predictions[target] = test_preds

print(f"\n Avg Validation MAPE: {total_mape / len(target_cols):.4f}")
predictions.to_csv("xgboost_optuna_submission.csv", index=False)
