```markdown
# Structured Data Classification & Feature Stability Pipeline

This repository contains a reproducible, end-to-end workflow for preprocessing tabular data, optimizing classification models, selecting features via stability/consensus analysis, and performing post-hoc interpretability evaluation. The pipeline is designed to strictly prevent data leakage, ensure full reproducibility, and provide reviewer-ready documentation.

## 📂 Execution Flow

The workflow is split into three sequential, interdependent scripts:

1. **`data_preprocessing_to_modeling.py`**  
   Data cleaning, stratified splitting (70/15/15), missing-value strategy selection (KNN vs. Iterative RF evaluated via masked RMSE), missingness indicator generation, Yeo-Johnson transformation for skewed variables, Z-score scaling, and artifact serialization.

2. **`model_stability_and_selection_framework.py`**  
   Hyperparameter tuning via Randomized Search CV, permutation importance (F1-degradation based), stability analysis across CV folds and random seeds, and consensus feature selection combining normalized importance, inverse coefficient of variation, and Kneedle-based selection frequency.

3. **`post_hoc_analysis_and_evaluation.py`**  
   Inverse transformation of scaled features to original units, Partial Dependence Plots (PDPs) for top predictors, and ROC/AUC evaluation across multiple model-subset combinations.

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib joblib kneed lightgbm xgboost catboost interpret pygam
```
*(Most dependencies are pre-installed in Google Colab or standard data science environments.)*

## 🚀 Usage

Execute the scripts **strictly in this order** to preserve data dependencies and artifact paths:

```bash
python data_preprocessing_to_modeling.py
python model_stability_and_selection_framework.py
python post_hoc_analysis_and_evaluation.py
```

### 🔧 Key Configuration Variables

| Variable | Description | File |
|:---|:---|:---|
| `SPEED_MODE` | `True` reduces iterations, folds, and subsampling limits for rapid testing. | `model_stability...` |
| `RUN_MODE` | Training regime: `"all"`, `"unbalanced_only"`, or `"balanced_only"`. | `model_stability...` |
| `MISSING_THRESH` | Column exclusion threshold for missing values (default: `0.30`). | `data_preprocessing...` |
| `SKEW_ABS_THRESH` | Absolute skewness threshold for Yeo-Johnson transformation (default: `1.0`). | `data_preprocessing...` |
| `INPUT_FILE` / `DATA_DIR` | Input dataset path and split directories (adjust for local/Colab environments). | All |

## 🔬 Methodological Notes

- **Leakage Prevention:** All transformers (imputer, scaler, Yeo-Johnson) are `fit` exclusively on the training split and `transform`ed on validation/test sets. Missing indicators are created **before** imputation.
- **Imbalance Handling:** The `unbalanced` regime applies class weighting (`class_weight='balanced'` or `scale_pos_weight`). The `balanced` regime uses random majority-class downsampling **only on training data** to preserve real-world test distributions.
- **Stable Feature Selection:** Instead of relying on a single metric, a consensus score is computed as:  
  `0.5 × Normalized Importance + 0.3 × Inverse CV% + 0.2 × Selection Frequency`.  
  The Kneedle algorithm determines optimal cutoffs for each subset.
- **Reproducibility:** `SEED=42` is fixed across stratified splits, CV folds, permutation repeats, and subsampling. All hyperparameters, transformation logs, and metadata are serialized in `_artifacts/`.

##  Reviewer & Reproduction Guide
1. Verify `INPUT_FILE` and `DATA_DIR` paths in the configuration blocks.
2. Run the three scripts sequentially.
3. For quick structural validation without full training, set `SPEED_MODE = True` in `model_stability_and_selection_framework.py`.
4. All methodological decisions, thresholds, and transformation parameters are logged in `_artifacts/pipeline_metadata.json` and script headers.

##  License

This code is provided for academic and research purposes. If adapted for publication or industrial deployment, please cite the source and preserve the reproducibility standards documented herein.
```
