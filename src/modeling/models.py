"""
Módulo de Modelos e Treinamento (src/modeling/models.py)
Definição dos algoritmos de classificação supervisionada (Regressão Logística Baseline,
Random Forest, XGBoost e LightGBM), integração estrita do pré-processamento via Pipeline,
Validação Cruzada Estratificada (5-Fold CV Zero Leakage) e persistência de modelos.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import RANDOM_STATE, N_SPLITS_CV, MODELS_DIR
from src.preprocessing.pipeline import EducationFeatureEngineer, build_preprocessor_pipeline


def get_candidate_models(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """
    Retorna o catálogo de modelos para avaliação comparativa.
    """
    models = {
        "Baseline_Logistic_Regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
    }
    return models


def create_full_pipeline(classifier: Any, feature_dict: dict) -> Pipeline:
    """
    Encapsula Feature Engineering, Preprocessamento e Classificador
    em um único objeto Pipeline Scikit-Learn (Zero Data Leakage).
    """
    fe = EducationFeatureEngineer()
    preprocessor = build_preprocessor_pipeline(feature_dict)
    
    return Pipeline([
        ("feature_engineer", fe),
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def evaluate_models_cross_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_dict: dict,
    n_splits: int = N_SPLITS_CV
) -> pd.DataFrame:
    """
    Executa Validação Cruzada Estratificada (Stratified 5-Fold CV) com Pipeline integrado
    para garantir ausência absoluta de vazamento de dados entre os folds de treino e validação.
    """
    print("\n" + "=" * 70)
    print(" 3. MODELAGEM & VALIDAÇÃO CRUZADA ESTRATIFICADA (5-FOLD CV ZERO LEAKAGE)")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
    }
    
    cv_results = []
    
    for name, model in models.items():
        print(f"[MODELING] Treinando e validando via Pipeline CV: {name:<30} ...", end="", flush=True)
        pipeline = create_full_pipeline(model, feature_dict)
        
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=None,
            return_train_score=False
        )
        
        row = {
            "Modelo": name,
            "ROC-AUC": f"{scores['test_roc_auc'].mean():.4f} (±{scores['test_roc_auc'].std():.3f})",
            "PR-AUC": f"{scores['test_pr_auc'].mean():.4f} (±{scores['test_pr_auc'].std():.3f})",
            "F1-Score": f"{scores['test_f1'].mean():.4f} (±{scores['test_f1'].std():.3f})",
            "Recall": f"{scores['test_recall'].mean():.4f} (±{scores['test_recall'].std():.3f})",
            "Precision": f"{scores['test_precision'].mean():.4f} (±{scores['test_precision'].std():.3f})",
            "Accuracy": f"{scores['test_accuracy'].mean():.4f} (±{scores['test_accuracy'].std():.3f})",
            "_raw_roc_auc_mean": scores["test_roc_auc"].mean(),
            "_raw_f1_mean": scores["test_f1"].mean(),
        }
        cv_results.append(row)
        print(" [OK]")
        
    df_results = pd.DataFrame(cv_results).sort_values(by="_raw_roc_auc_mean", ascending=False)
    print("\n--- Resultados da Validação Cruzada Estratificada (Média ± Desvio Padrão) ---")
    print(df_results.drop(columns=["_raw_roc_auc_mean", "_raw_f1_mean"]).to_string(index=False))
    
    return df_results


def fit_and_save_all_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_dict: dict
) -> Dict[str, Pipeline]:
    """
    Ajusta todos os pipelines de modelos no conjunto de treino completo e salva os artefatos binários.
    """
    trained_pipelines = {}
    print("\n[MODELING] Ajustando Pipelines completos no conjunto de treino (Treino 80%)...")
    
    for name, model in models.items():
        pipeline = create_full_pipeline(model, feature_dict)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Salvar pipeline treinado completo
        model_path = MODELS_DIR / f"{name.lower()}.joblib"
        joblib.dump(pipeline, model_path)
        
    print(f"[MODELING] Todos os {len(trained_pipelines)} pipelines treinados e salvos em: {MODELS_DIR}")
    return trained_pipelines

