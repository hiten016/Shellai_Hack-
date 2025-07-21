import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_solution.csv")

blend_cols = [col for col in train.columns if 'fraction' in col.lower()]
component_cols = [col for col in train.columns if 'Component' in col and 'Property' in col]
target_cols = [f'BlendProperty{i}' for i in range(1, 11)]

def add_weighted_features(df):
    df = df.copy()
    for i in range(1, 11):
        weighted_sum = 0
        for j in range(1, 6):
            weighted_sum += df[blend_cols[j - 1]] * df[f"Component{j}_Property{i}"]
        df[f"Weighted_Property{i}"] = weighted_sum
    return df

def add_nonlinear_features(df):
    df = df.copy()
    for col in blend_cols + component_cols:
        df[f"{col}_squared"] = df[col] ** 2
    for i in range(len(blend_cols)):
        for j in range(len(component_cols)):
            df[f"{blend_cols[i]}x{component_cols[j]}"] = df[blend_cols[i]] * df[component_cols[j]]
    for col in blend_cols + component_cols:
        df[f"log_{col}"] = np.log1p(df[col])
    return df

train = add_weighted_features(train)
test = add_weighted_features(test)
train = add_nonlinear_features(train)
test = add_nonlinear_features(test)

weighted_cols = [f"Weighted_Property{i}" for i in range(1, 11)]
nonlinear_cols = [col for col in train.columns if 'squared' in col or 'x' in col or 'log' in col]
all_features = blend_cols + component_cols + weighted_cols + nonlinear_cols

train = train[all_features + target_cols]
train.fillna(train.median(numeric_only=True), inplace=True)
test = test[all_features]
test.fillna(test.median(numeric_only=True), inplace=True)

X = train[all_features]
y = train[target_cols]
X_test = test[all_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb = MultiOutputRegressor(XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42
), n_jobs=-1)

catboost = MultiOutputRegressor(CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=5, verbose=0, random_state=42
), n_jobs=-1)

lgbm = MultiOutputRegressor(LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, random_state=42
), n_jobs=-1)

oof_xgb = np.zeros_like(y)
oof_cat = np.zeros_like(y)
oof_lgbm = np.zeros_like(y)
test_preds_xgb = np.zeros((X_test.shape[0], y.shape[1]))
test_preds_cat = np.zeros((X_test.shape[0], y.shape[1]))
test_preds_lgbm = np.zeros((X_test.shape[0], y.shape[1]))

print("🔁 Training K-Folds...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"📂 Fold {fold + 1}")
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb.fit(X_tr, y_tr)
    catboost.fit(X_tr, y_tr)
    lgbm.fit(X_tr, y_tr)

    oof_xgb[val_idx] = xgb.predict(X_val)
    oof_cat[val_idx] = catboost.predict(X_val)
    oof_lgbm[val_idx] = lgbm.predict(X_val)

    test_preds_xgb += xgb.predict(X_test_scaled) / kf.get_n_splits()
    test_preds_cat += catboost.predict(X_test_scaled) / kf.get_n_splits()
    test_preds_lgbm += lgbm.predict(X_test_scaled) / kf.get_n_splits()

meta_X = np.concatenate([oof_xgb, oof_cat, oof_lgbm], axis=1)
meta_test_X = np.concatenate([test_preds_xgb, test_preds_cat, test_preds_lgbm], axis=1)

ridge = MultiOutputRegressor(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), n_jobs=-1)
lasso = MultiOutputRegressor(make_pipeline(StandardScaler(), Lasso(alpha=0.001)), n_jobs=-1)
elastic = MultiOutputRegressor(make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5)), n_jobs=-1)

ridge.fit(meta_X, y)
lasso.fit(meta_X, y)
elastic.fit(meta_X, y)

ridge_preds = ridge.predict(meta_X)
lasso_preds = lasso.predict(meta_X)
elastic_preds = elastic.predict(meta_X)

val_preds = (ridge_preds + lasso_preds + elastic_preds) / 3
val_mape = mean_absolute_percentage_error(y, val_preds)
print(f"\n📉 Final Validation MAPE (Meta: Ridge + Lasso + ElasticNet): {val_mape:.4f}")

ridge_test = ridge.predict(meta_test_X)
lasso_test = lasso.predict(meta_test_X)
elastic_test = elastic.predict(meta_test_X)

final_test_preds = (ridge_test + lasso_test + elastic_test) / 3

submission = sample_submission.copy()
submission.iloc[:, 1:] = final_test_preds
submission.to_csv("/kaggle/working/ridge_lasso_elasticnet_lgbm_stack_submission.csv", index=False)
print("✅ Submission saved as 'ridge_lasso_elasticnet_lgbm_stack_submission.csv'")
