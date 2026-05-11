"""Kanoniczne nazwy kolumn encji — zgodne z kluczami w plikach YAML."""

# Muszą być identyczne z kluczami w config/biomarkers.yaml
ALL_BIOMARKER_COLUMNS: list[str] = [
    "CRP",
    "troponin",
    "cholesterol",
    "non_hdl",
    "HbA1c",
    "IL-6",
    "glucose",
    "BNP",
    "ferritin",
    "albumin",
    "creatinine",
]

# Muszą być identyczne z kluczami w config/ml_methods.yaml
ALL_ML_METHOD_COLUMNS: list[str] = [
    "linear_regression",
    "cox_model",
    "logistic_regression",
    "random_forest",
    "SVM",
    "neural_network",
    "deep_learning",
    "transformer",
    "XGBoost",
    "gradient_boosting",
    "lightgbm",
    "catboost",
    "naive_bayes",
    "decision_tree",
    "clustering",
]
