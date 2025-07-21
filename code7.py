# 📦 Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import NuSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# 📁 Load Data (local folder version)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_solution.csv")

# 🎯 Columns
blend_cols = [col for col in train.columns if 'fraction' in col.lower()]
component_cols = [col for col in train.columns if 'Component' in col and 'Property' in col]
target_cols = [f'BlendProperty{i}' for i in range(1, 11)]

# ⚙ Weighted Average Features
def add_weighted_features(df):
    df = df.copy()
    for i in range(1, 11):
        weighted_sum = 0
        for j in range(1, 6):
            weighted_sum += df[blend_cols[j - 1]] * df[f"Component{j}_Property{i}"]
        df[f"Weighted_Property{i}"] = weighted_sum
    return df

# 🔁 Nonlinear Features
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

# 🧪 Feature Engineering
train = add_weighted_features(train)
test = add_weighted_features(test)
train = add_nonlinear_features(train)
test = add_nonlinear_features(test)

# 📊 Features
weighted_cols = [f"Weighted_Property{i}" for i in range(1, 11)]
nonlinear_cols = [col for col in train.columns if 'squared' in col or '_x' in col or 'log_' in col]
all_features = blend_cols + component_cols + weighted_cols + nonlinear_cols

# 🧼 Clean Data
train = train[all_features + target_cols]
train.fillna(train.median(numeric_only=True), inplace=True)
test = test[all_features]
test.fillna(test.median(numeric_only=True), inplace=True)

X = train[all_features]
y = train[target_cols]
X_test = test[all_features]

# 📐 Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 🔁 KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ⚙️ Base Models
xgb = MultiOutputRegressor(XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42
), n_jobs=-1)

catboost = MultiOutputRegressor(CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=5, verbose=0, random_state=42
), n_jobs=-1)

# 🧠 Stacking storage
oof_xgb = np.zeros_like(y)
oof_cat = np.zeros_like(y)
test_preds_xgb = np.zeros((X_test.shape[0], y.shape[1]))
test_preds_cat = np.zeros((X_test.shape[0], y.shape[1]))

print("🔁 Training K-Folds...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"📂 Fold {fold + 1}")
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb.fit(X_tr, y_tr)
    catboost.fit(X_tr, y_tr)

    oof_xgb[val_idx] = xgb.predict(X_val)
    oof_cat[val_idx] = catboost.predict(X_val)

    test_preds_xgb += xgb.predict(X_test_scaled) / kf.get_n_splits()
    test_preds_cat += catboost.predict(X_test_scaled) / kf.get_n_splits()

# 🤖 Meta Model with NuSVR
meta_X = np.concatenate([oof_xgb, oof_cat], axis=1)
meta_test_X = np.concatenate([test_preds_xgb, test_preds_cat], axis=1)

meta_model = MultiOutputRegressor(make_pipeline(
    StandardScaler(),
    NuSVR(kernel='rbf', nu=0.5, C=10, gamma='scale')
), n_jobs=-1)

meta_model.fit(meta_X, y)
val_preds = meta_model.predict(meta_X)
val_mape = mean_absolute_percentage_error(y, val_preds)
print(f"\n📉 Final Validation MAPE (Meta: NuSVR RBF Kernel): {val_mape:.4f}")

# 📤 Submission
final_test_preds = meta_model.predict(meta_test_X)
submission = sample_submission.copy()
submission.iloc[:, 1:] = final_test_preds
submission.to_csv("nusvr_rbf_meta_submission.csv", index=False)
print("✅ Submission saved as 'new.csv'")
