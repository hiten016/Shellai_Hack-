import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

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

predictions = pd.DataFrame()
predictions["ID"] = test["ID"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
total_mape = 0

for target in target_cols:
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]
        model = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.7,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / kf.n_splits
    mape = mean_absolute_percentage_error(y[target], oof)
    total_mape += mape
    print(f"{target} MAPE: {mape:.4f}")
    predictions[target] = test_preds

print(f"\n\U0001F50D Avg Validation MAPE: {total_mape / 10:.4f}")

predictions.to_csv("xgboost_cv_submission.csv", index=False)

