# ===========================================================================
# POST-HOC ANALYSIS & MODEL EVALUATION SCRIPT
# ===========================================================================

import os
import re
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# 1. CONFIGURATION & PATHS
# ===========================================================================
MODEL_DIR   = "/content"
TEST_CSV    = os.path.join(MODEL_DIR, "test.csv")
SCALER_CSV  = os.path.join(MODEL_DIR, "scaler_params_forced.csv")
YJ_JSON     = os.path.join(MODEL_DIR, "yeojohnson_lambdas.json")
OUT_DIR     = os.path.join(MODEL_DIR, "analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = "Conflict"
TOP4_FEATS = ["Mean Temperature", "Internet", "Precipitation", "Political stability"]
FEATURE_UNITS = {
    "Mean Temperature": "°C",
    "Internet": "% of population",
    "Precipitation": "mm/year",
    "Political stability": "WGI score (-2.5 to +2.5)",
}

# Models to evaluate for ROC analysis
# Format: (regime, feature_subset, model_tag, model_path)
ROC_MODEL_SPECS = [
    ("unbalanced", "all19", "lightgbm",      os.path.join(MODEL_DIR, "lightgbm.joblib")),
    ("unbalanced", "top4",  "random_forest", os.path.join(MODEL_DIR, "random_forest.joblib")),
    ("unbalanced", "top9",  "catboost",      os.path.join(MODEL_DIR, "catboost.joblib")),
]

# ===========================================================================
# 2. INVERSE TRANSFORMATION UTILITIES
# ===========================================================================
def _normalize_name(s: str) -> str:
    """Normalizes feature names for robust cross-file matching."""
    s = str(s).strip().lower()
    s = re.sub(r"[\s_\-]+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

FEATURE_SYNONYMS = {
    "precipitation": ["perception", "precip", "prcp", "rain", "rainfall"],
    "meantemperature": ["meantemp", "tempmean", "temperaturemean"],
    "internet": ["internetusers", "internetpenetration", "internetpercent"],
    "politicalstability": ["political_stability", "political-stability"],
}

def _get_candidates(name: str) -> list:
    """Generates possible normalized aliases for parameter lookup."""
    k = _normalize_name(name)
    cands = [k] + FEATURE_SYNONYMS.get(k, [])
    cands += [f"{k}yj", f"{k}scaled", f"{k}__scaled", f"{k}__yj"]
    return list(dict.fromkeys([_normalize_name(x) for x in cands]))

def load_scaler_params(path: str):
    """Loads mean/std parameters for StandardScaler inversion."""
    if not os.path.exists(path): return {}, {}
    sc = pd.read_csv(path)
    sc["feature_norm"] = (sc["feature"].astype(str).str.lower()
                          .str.replace(r"[\s_\-]+", "", regex=True)
                          .str.replace(r"[^a-z0-9]+", "", regex=True))
    return dict(zip(sc["feature_norm"], sc["mean"])), dict(zip(sc["feature_norm"], sc["std"]))

def load_yj_lambdas(path: str):
    """Loads Yeo-Johnson lambda parameters for inverse transformation."""
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("lambdas", data)
    return {_normalize_name(k): float(v) for k, v in raw.items()}

def inverse_yeojohnson(y: np.ndarray, lam: float) -> np.ndarray:
    """Computes the inverse Yeo-Johnson transformation."""
    y = np.asarray(y, dtype=float)
    x = np.empty_like(y)
    pos, neg = y >= 0, ~pos
    
    if abs(lam) < 1e-9:
        x[pos] = np.exp(y[pos]) - 1.0
    else:
        x[pos] = np.power(lam * y[pos] + 1.0, 1.0 / lam) - 1.0
        
    if abs(lam - 2.0) < 1e-9:
        x[neg] = 1.0 - np.exp(-y[neg])
    else:
        x[neg] = 1.0 - np.power(1.0 - (2.0 - lam) * y[neg], 1.0 / (2.0 - lam))
    return x

def restore_original_values(x_scaled: np.ndarray, mean: float, std: float, lam: float) -> np.ndarray:
    """Applies inverse StandardScaler followed by inverse Yeo-Johnson."""
    x_pre_yj = x_scaled * std + mean
    return inverse_yeojohnson(x_pre_yj, lam) if abs(lam - 1.0) > 1e-9 else x_pre_yj

# ===========================================================================
# 3. PARTIAL DEPENDENCE PLOT (PDP) GENERATOR
# ===========================================================================
def _extract_pdp_data(estimator, X: pd.DataFrame, feat_idx: int):
    """Computes PDP grid and average values with sklearn version compatibility."""
    res = partial_dependence(estimator, X, [feat_idx], kind="average")
    if hasattr(res, "keys"):  # sklearn >= 1.2
        xg = res.get("grid_values", res.get("values"))[0]
        ya = res["average"][0]
    else:  # sklearn < 1.2
        averages, values = res
        xg, ya = values[0], averages[0]
    return np.asarray(xg), np.asarray(ya)

def generate_pdp(model_path: str, features: list, units: dict, out_path: str):
    """Generates and saves a 2x2 subplot figure of PDPs in original units."""
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found for PDP: {model_path}"); return
        
    model = joblib.load(model_path)
    X_test = pd.read_csv(TEST_CSV)[features].copy()
    mean_map, std_map = load_scaler_params(SCALER_CSV)
    lambdas = load_yj_lambdas(YJ_JSON)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        x_scaled, y_avg = _extract_pdp_data(model, X_test, i)
        mu, sd = mean_map.get(_normalize_name(feat)), std_map.get(_normalize_name(feat))
        lam = next((lambdas[k] for k in _get_candidates(feat) if k in lambdas), 1.0)

        x_raw = restore_original_values(x_scaled, mu, sd, lam) if mu is not None and sd is not None else x_scaled

        axes[i].plot(x_raw, y_avg, linewidth=2, color="#1f77b4")
        axes[i].set_title(feat, fontsize=12, fontweight="bold")
        axes[i].set_xlabel(f"{feat} [{units.get(feat, 'original units')}]", fontsize=10)
        axes[i].set_ylabel("Partial dependence (P(conflict = 1))", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("Partial Dependence Plots — RandomForest (Top 4 Features)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] PDP figure saved to: {out_path}")
    plt.close(fig)

# ===========================================================================
# 4. ROC CURVE EVALUATION & PLOT GENERATOR
# ===========================================================================
def _get_predict_scores(model, df: pd.DataFrame) -> np.ndarray:
    """Extracts positive-class probabilities or decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(df)
        return 1.0 / (1.0 + np.exp(-scores))  # Sigmoid normalization for ROC
    else:
        raise ValueError("Model must support predict_proba or decision_function.")

def evaluate_and_plot_roc(model_specs: list, df_test: pd.DataFrame, target_col: str, out_dir: str):
    """Generates combined and individual ROC curves with AUC metrics."""
    y_true = df_test[target_col].astype(int).values
    results = []
    
    plt.figure(figsize=(6, 6))
    for regime, subset, tag, path in model_specs:
        if not os.path.exists(path):
            print(f"[WARN] Skipping missing model: {path}")
            continue
            
        pipe = joblib.load(path)
        y_score = _get_predict_scores(pipe, df_test)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        results.append({"regime": regime, "subset": subset, "tag": tag, "auc": auc})
        plt.plot(fpr, tpr, lw=2, label=f"{subset}-{tag} (AUC={auc:.3f})")

        # Individual ROC plot
        fig_ind, ax_ind = plt.subplots(figsize=(6, 6))
        ax_ind.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
        ax_ind.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax_ind.set_xlabel("False Positive Rate")
        ax_ind.set_ylabel("True Positive Rate")
        ax_ind.set_title(f"ROC — Test | {subset}-{tag}")
        ax_ind.legend(loc="lower right")
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", f"{regime}_{subset}_{tag}").strip("_")
        fig_ind.savefig(os.path.join(out_dir, f"ROC_{safe_name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_ind)

    # Combined ROC plot
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Test Set (Model Comparison)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ROC_compare_unbalanced.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[SUMMARY] AUC Scores:")
    for r in results:
        print(f"  {r['regime']:>10} | {r['subset']:>5} | {r['tag']:<14} -> AUC={r['auc']:.4f}")

# ===========================================================================
# 5. MAIN EXECUTION BLOCK
# ===========================================================================
if __name__ == "__main__":
    print(" STARTING POST-HOC ANALYSIS & EVALUATION")

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test dataset not found at {TEST_CSV}")
        
    df_test_full = pd.read_csv(TEST_CSV)

    # 1. Generate Partial Dependence Plots
    rf_path = next((p for _, _, t, p in ROC_MODEL_SPECS if t == "random_forest"), None)
    if rf_path and os.path.exists(rf_path):
        generate_pdp(rf_path, TOP4_FEATS, FEATURE_UNITS, os.path.join(OUT_DIR, "PDP_RandomForest_top4.png"))
    else:
        print("[SKIP] Random Forest model not found for PDP generation.")

    # 2. Evaluate and Plot ROC Curves
    evaluate_and_plot_roc(ROC_MODEL_SPECS, df_test_full, TARGET_COL, OUT_DIR)

    print(" ALL POST-HOC ANALYSES COMPLETED. OUTPUTS SAVED TO:", OUT_DIR)