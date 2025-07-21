import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_solution.csv")

target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
feature_cols = train.columns.difference(target_cols).tolist()

X = train[feature_cols]
y = train[target_cols]
X_test = test[feature_cols]

predictions = pd.DataFrame()
for target in target_cols:
    model = XGBRegressor(random_state=42, n_jobs=-1)
    model.fit(X, y[target])
    preds = model.predict(X_test)
    predictions[target] = preds

submission = pd.DataFrame()
submission["ID"] = test["ID"]
submission[target_cols] = predictions
submission.to_csv("xgboost_submission.csv", index=False)
