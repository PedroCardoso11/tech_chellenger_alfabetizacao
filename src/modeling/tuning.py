"""
Módulo de Otimização de Hiperparâmetros (src/modeling/tuning.py)
Implementa busca bayesiana via Optuna integrada com Pipeline Scikit-Learn
e Validação Cruzada Estratificada 5-fold (Zero Data Leakage).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

from src.config import RANDOM_STATE, N_SPLITS_CV, N_OPTUNA_TRIALS, MODELS_DIR
from src.modeling.models import create_full_pipeline

# Suprimir logs verbosos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_lightgbm_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_dict: dict,
    n_trials: int = N_OPTUNA_TRIALS
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Otimiza hiperparâmetros do LightGBM usando Optuna com validação cruzada 5-fold sobre o Pipeline completo.
    
    Returns:
        best_pipeline: Pipeline com pré-processador e classificador ajustado com hiperparâmetros ótimos.
        best_params: Dicionário com os hiperparâmetros ótimos.
        best_score: Melhor ROC-AUC médio obtido na validação cruzada.
    """
    print("\n" + "=" * 70)
    print(" 3.1 OTIMIZAÇÃO DE HIPERPARÂMETROS VIA OPTUNA (PIPELINE BAYESIAN SEARCH)")
    print("=" * 70)
    print(f"[TUNING] Iniciando busca bayesiana ({n_trials} trials) com Stratified 5-Fold CV...")
    
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1
        }
        
        clf = LGBMClassifier(**params)
        pipeline = create_full_pipeline(clf, feature_dict)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=None)
        return scores.mean()
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"[TUNING] Otimização Concluída!")
    print(f"  - Melhor ROC-AUC CV: {best_score:.4f}")
    print("  - Hiperparâmetros Ótimos:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"      * {k:<20}: {v:.5f}")
        else:
            print(f"      * {k:<20}: {v}")
            
    # Treinar pipeline final otimizado no conjunto completo de treino
    best_params_full = {
        **best_params,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }
    best_clf = LGBMClassifier(**best_params_full)
    best_pipeline = create_full_pipeline(best_clf, feature_dict)
    best_pipeline.fit(X_train, y_train)
    
    # Salvar modelo otimizado
    model_path = MODELS_DIR / "lightgbm_optimized.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"[TUNING] Pipeline otimizado salvo em: {model_path}")
    
    return best_pipeline, best_params_full, best_score

