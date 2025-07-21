import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_solution.csv")

# Separate features and targets
X = train.drop(columns=[col for col in train.columns if "BlendProperty" in col])
y = train[[f"BlendProperty{i}" for i in range(1, 11)]]
X_test = test.drop(columns=["ID"])

# Feature Engineering
def add_features(df):
    df = df.copy()
    for i in range(1, 6):
        for j in range(1, 11):
            df[f'Weighted_Property{j}_C{i}'] = df[f'Component{i}_fraction'] * df[f'Component{i}_Property{j}']
    for j in range(1, 11):
        cols = [f'Weighted_Property{j}_C{i}' for i in range(1, 6)]
        df[f'WeightedAvg_Property{j}'] = df[cols].sum(axis=1)
    for col in df.select_dtypes(include=np.number).columns:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    col_list = df.select_dtypes(include=[np.number]).columns[:10]
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            df[f'{col_list[i]}_x_{col_list[j]}'] = df[col_list[i]] * df[col_list[j]]
    return df

X_fe = add_features(X).select_dtypes(include=np.number)
X_test_fe = add_features(X_test).select_dtypes(include=np.number)

# Models
xgb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "random_state": 42}
cat_params = {"n_estimators": 100, "depth": 5, "learning_rate": 0.05, "random_state": 42, "verbose": 0}
svr_model = lambda: make_pipeline(StandardScaler(), SVR(kernel='linear', C=0.5))

# Stacking across all 10 targets
submission = pd.DataFrame()
submission["ID"] = test["ID"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i in range(10):
    print(f"Training for BlendProperty{i+1}")
    blend_preds_train = np.zeros((X_fe.shape[0], 2))
    blend_preds_test = np.zeros((X_test_fe.shape[0], 2))
    y_col = y.iloc[:, i]
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_fe)):
        X_tr, X_val = X_fe.iloc[tr_idx], X_fe.iloc[val_idx]
        y_tr, y_val = y_col.iloc[tr_idx], y_col.iloc[val_idx]

        xgb = XGBRegressor(**xgb_params)
        cat = CatBoostRegressor(**cat_params)

        xgb.fit(X_tr, y_tr)
        cat.fit(X_tr, y_tr)

        blend_preds_train[val_idx, 0] = xgb.predict(X_val)
        blend_preds_train[val_idx, 1] = cat.predict(X_val)

        blend_preds_test[:, 0] += xgb.predict(X_test_fe) / kf.n_splits
        blend_preds_test[:, 1] += cat.predict(X_test_fe) / kf.n_splits

    # Train meta-model
    svr = svr_model()
    svr.fit(blend_preds_train, y_col)
    final_preds = svr.predict(blend_preds_test)

    # Save predictions
    submission[f"BlendProperty{i+1}"] = final_preds

# Save to CSV
submission.to_csv("stacked_submission.csv", index=False)
print("✅ Submission file saved as 'stacked_submission.csv'")
