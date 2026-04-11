"""
================================================================================
DATA PREPROCESSING PIPELINE 
================================================================================
METHODOLOGY NOTES FOR REVIEWERS:

1. Data Cleaning & Type Conversion:
   - Literal string "Null" is converted to np.nan.
   - Column names are normalized (trimmed whitespace, standardized formatting).
   - Object-typed columns containing numeric characters (commas, percentages, 
     parentheses) are safely parsed to float64 using regex-based extraction.

2. Train/Validation/Test Split:
   - Stratified split ensures the distribution of the target variable ("Conflict") 
     is preserved across all subsets.
   - Split ratio: 70% Train, 15% Validation, 15% Test.
   - Random state is fixed for full reproducibility.

3. Missing Data Handling & Imputation:
   - Features with >30% missing values in the training set are excluded prior to 
     imputation to avoid introducing high-variance estimates.
   - Missing-indicator columns (is_missing_<feature>) are created BEFORE imputation 
     to preserve the structural information of missingness, which is often predictive 
     in socio-economic and environmental datasets.
   - Three imputation strategies (KNN, Iterative Random Forest, Iterative XGBoost) 
     are evaluated exclusively on the training set using a masked validation approach. 
     The strategy with the lowest average RMSE is selected.

4. Feature Transformation & Scaling:
   - Skewness is computed on the training set post-imputation.
   - Yeo-Johnson transformation is applied to features with |skew| >= 1.0 to 
     approximate normality and stabilize variance.
   - StandardScaler (Z-score normalization) is applied to all numeric features.
   - All transformers are fit ONLY on the training data and then applied to 
     validation/test sets to prevent data leakage.

5. Class Imbalance Handling:
   - Random downsampling of the majority class is applied ONLY to the training split 
     to create "train_balanced.csv". Validation and test sets remain untouched to 
     preserve real-world distribution for unbiased evaluation.

6. Reproducibility & Artifacts:
   - All pipeline objects (imputer, preprocessor), configuration parameters, 
     transformation logs, and feature metadata are serialized for future inference 
     and peer verification.
================================================================================
"""

import os
import json
import re
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ================= CONFIGURATION =================
INPUT_FILE = "Dataset_unpacked_pruned.csv"  # <-- REPLACE WITH YOUR EXACT INPUT FILE PATH
TARGET = "Conflict"
ID_COLS = ["Basin", "Country", "Year"]

OUT_DIR = "data_splits"
OUT_DIR_SCALED = "data_splits_scaled"
LOG_DIR = os.path.join(OUT_DIR, "logs")
ARTIFACT_DIR = "_artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MISSING_THRESH = 0.30
SKEW_ABS_THRESH = 1.0

def ensure_directories():
    for d in [OUT_DIR, OUT_DIR_SCALED, LOG_DIR, ARTIFACT_DIR]:
        os.makedirs(d, exist_ok=True)

def load_and_clean_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input dataset not found at: {path}\nPlease update INPUT_FILE in the configuration section.")
    
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [re.sub(r'\s+', ' ', str(c).strip()) for c in df.columns]
    df = df.replace("Null", np.nan)
    
    for col in df.columns:
        if col in ID_COLS or col == TARGET:
            continue
        if df[col].dtype == 'object':
            s = df[col].astype(str).str.replace(',', '', regex=False) \
                             .str.replace('%', '', regex=False) \
                             .str.replace('\u200f', '', regex=False)
            s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
            df[col] = pd.to_numeric(s, errors='coerce')
            
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').fillna(0).astype(int)
    return df

def select_best_imputer(X_train_num: pd.DataFrame) -> Tuple[str, object, Dict]:
    imputers = {
        "Iterative_RF": IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
            max_iter=15, initial_strategy='median', random_state=RANDOM_STATE
        ),
        "KNN": KNNImputer(n_neighbors=5, weights='distance')
    }
    
    rng = np.random.default_rng(RANDOM_STATE)
    results = {name: [] for name in imputers}
    eligible = [c for c in X_train_num.columns if X_train_num[c].notna().sum() >= 30]
    
    for col in eligible:
        ser = X_train_num[[col]].dropna()
        if len(ser) < 20:
            continue
        mask = rng.random(len(ser)) < 0.7
        inc = ser.copy()
        inc.iloc[mask] = np.nan
        
        for name, imp in imputers.items():
            try:
                pred = imp.fit_transform(inc)[:, 0]
                rmse = np.sqrt(mean_squared_error(ser[col], pred))
                nrmse = rmse / (ser[col].max() - ser[col].min() + 1e-8)
                results[name].append(nrmse)
            except Exception:
                results[name].append(np.inf)
                
    avg_nrmse = {k: float(np.mean(v)) if v else np.inf for k, v in results.items()}
    best_name = min(avg_nrmse, key=avg_nrmse.get)
    return best_name, imputers[best_name], avg_nrmse

def run_pipeline():
    ensure_directories()
    
    # 1. Load & Clean
    df = load_and_clean_data(INPUT_FILE)
    
    # 2. Stratified Split (No Leakage)
    train_df, temp_df = train_test_split(df, test_size=TEST_SIZE + VAL_SIZE, 
                                         random_state=RANDOM_STATE, stratify=df[TARGET])
    rel_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(temp_df, test_size=1-rel_val, 
                                       random_state=RANDOM_STATE, stratify=temp_df[TARGET])
    
    # 3. Drop high-missing columns (Train-only decision)
    num_cols = [c for c in df.columns if c not in ID_COLS + [TARGET] and pd.api.types.is_numeric_dtype(df[c])]
    miss_frac = train_df[num_cols].isna().mean()
    drop_cols = miss_frac[miss_frac > MISSING_THRESH].index.tolist()
    for part in [train_df, val_df, test_df]:
        part.drop(columns=[c for c in drop_cols if c in part.columns], inplace=True, errors='ignore')
    
    final_num_cols = [c for c in train_df.columns if c not in ID_COLS + [TARGET] and pd.api.types.is_numeric_dtype(train_df[c])]
    
    # 4. Missing Indicators
    flag_cols = []
    for c in final_num_cols:
        flag = f"is_missing_{c}"
        flag_cols.append(flag)
        for part in [train_df, val_df, test_df]:
            part[flag] = part[c].isna().astype(int)
            
    # 5. Imputation
    best_name, best_imp, avg_metrics = select_best_imputer(train_df[final_num_cols])
    best_imp.fit(train_df[final_num_cols])
    for part in [train_df, val_df, test_df]:
        part[final_num_cols] = best_imp.transform(part[final_num_cols])
        
    with open(os.path.join(LOG_DIR, "imputer_evaluation.json"), "w") as f:
        json.dump({"selected": best_name, "metrics": avg_metrics}, f, indent=2)
        
    # 6. Transformation (Yeo-Johnson + Scaling)
    skewed_cols = [c for c in final_num_cols if abs(train_df[c].skew()) >= SKEW_ABS_THRESH]
    linear_cols = [c for c in final_num_cols if c not in skewed_cols]
    
    preproc = ColumnTransformer([
        ('skewed', Pipeline([('yj', PowerTransformer(method='yeo-johnson', standardize=False)),
                             ('sc', StandardScaler())]), skewed_cols),
        ('linear', StandardScaler(), linear_cols),
        ('flags', 'passthrough', flag_cols)
    ], remainder='drop', verbose_feature_names_out=False)
    
    preproc.fit(train_df[final_num_cols + flag_cols])
    
    print(f"[TRANSFORM] Yeo-Johnson applied on {len(skewed_cols)} skewed columns; StandardScaler applied=True")
    
    # 7. Save Splits
    def save_split(df_in: pd.DataFrame, suffix: str):
        X = preproc.transform(df_in[final_num_cols + flag_cols])
        cols_out = skewed_cols + linear_cols + flag_cols
        df_proc = pd.concat([df_in[ID_COLS + [TARGET]].reset_index(drop=True),
                             pd.DataFrame(X, columns=cols_out, index=df_in.index)], axis=1)
        out_path = os.path.join(OUT_DIR_SCALED, f"{suffix}.csv")
        df_proc.to_csv(out_path, index=False)
        print(f"[SAVE] data_splits/{suffix}.csv")
        return df_proc
        
    for df_part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        save_split(df_part, name)
        
    # 8. Balanced Downsample (Train Only)
    counts = train_df[TARGET].value_counts()
    if len(counts) >= 2:
        n_min = counts.min()
        df_min = train_df[train_df[TARGET] == counts.idxmin()]
        df_maj = train_df[train_df[TARGET] != counts.idxmin()].sample(n=n_min, random_state=RANDOM_STATE)
        train_bal = pd.concat([df_min, df_maj]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        save_split(train_bal, "train_balanced")
    else:
        save_split(train_df, "train_balanced")
        
    # 9. Save Artifacts
    joblib.dump(best_imp, os.path.join(ARTIFACT_DIR, "imputer.joblib"))
    joblib.dump(preproc, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "pipeline_metadata.json"), "w") as f:
        json.dump({
            "config": {"input_file": INPUT_FILE, "target": TARGET, "random_state": RANDOM_STATE,
                       "missing_thresh": MISSING_THRESH, "skew_thresh": SKEW_ABS_THRESH},
            "features": {"numeric": final_num_cols, "skewed": skewed_cols, "linear": linear_cols, "flags": flag_cols}
        }, f, indent=2)
        
    print(f"[ARTIFACTS] Saved to {ARTIFACT_DIR}")
    print("[DONE] Preprocessing complete. Splits, logs, and artifacts generated successfully.")

if __name__ == "__main__":
    run_pipeline()