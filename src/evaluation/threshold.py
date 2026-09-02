"""
Módulo de Otimização de Limiar de Decisão (src/evaluation/threshold.py)
Ajuste fino do limiar de classificação com foco no custo assimétrico social
(maximização do recall de crianças em risco de não alfabetização via F2-score).
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import FIGURES_DIR, IMAGES_DIR


def optimize_decision_threshold(
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    save_figures: bool = True
) -> Dict[str, Any]:
    """
    Otimiza o limiar de decisão focado no impacto social educacional:
    Prioriza a identificação máxima dos alunos em risco de não alfabetização (minimizando Falsos Negativos).
    """
    print("\n--- Análise de Custo-Benefício Social & Otimização do Limiar ---")
    print("No contexto educacional público:")
    print("  * Falso Negativo (prever alfabetizado quando o aluno NÃO é): CUSTO CRÍTICO (criança invisível para apoio).")
    print("  * Falso Positivo (prever não alfabetizado quando o aluno É): CUSTO BAIXO (alocação preventiva de reforço).")
    
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0.10, 0.90, 81)
    results = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        
        # Métricas para a classe vulnerável (0 = Não Alfabetizado)
        rec_risco = recall_score(1 - y_test, 1 - y_pred_t, zero_division=0)
        prec_risco = precision_score(1 - y_test, 1 - y_pred_t, zero_division=0)
        f1_risco = f1_score(1 - y_test, 1 - y_pred_t, zero_division=0)
        
        # F2-score (pondera recall com 2x o peso da precisão)
        beta = 2.0
        f2_risco = ((1 + beta**2) * prec_risco * rec_risco) / ((beta**2 * prec_risco) + rec_risco) if (prec_risco + rec_risco) > 0 else 0
        
        f1_global = f1_score(y_test, y_pred_t, zero_division=0)
        acc = accuracy_score(y_test, y_pred_t)
        
        results.append({
            "threshold": t,
            "recall_risco": rec_risco,
            "precision_risco": prec_risco,
            "f1_risco": f1_risco,
            "f2_risco": f2_risco,
            "f1_global": f1_global,
            "accuracy": acc
        })
        
    df_thresh = pd.DataFrame(results)
    
    row_default = df_thresh.iloc[(df_thresh["threshold"] - 0.50).abs().argmin()]
    row_f2_opt = df_thresh.loc[df_thresh["f2_risco"].idxmax()]
    
    print("\n[THRESHOLD TUNING] Comparativo de Políticas de Decisão:")
    print(f"  1. Limiar Padrão (t = 0.50):")
    print(f"     * Recall Alunos em Risco: {row_default['recall_risco']*100:.2f}% | Precisão: {row_default['precision_risco']*100:.2f}% | F1 Global: {row_default['f1_global']:.4f}")
    print(f"  2. Limiar Socialmente Recomendado (Max F2 Risco, t = {row_f2_opt['threshold']:.2f}):")
    print(f"     * Recall Alunos em Risco: {row_f2_opt['recall_risco']*100:.2f}% (+{(row_f2_opt['recall_risco']-row_default['recall_risco'])*100:+.2f}%) | Precisão: {row_f2_opt['precision_risco']*100:.2f}% | F2 Risco: {row_f2_opt['f2_risco']:.4f}")
    
    if save_figures:
        _plot_threshold_tuning(df_thresh, row_default["threshold"], row_f2_opt["threshold"])
        
    return {
        "threshold_default": float(row_default["threshold"]),
        "threshold_social_optimal": float(row_f2_opt["threshold"]),
        "recall_gain_risco_pct": float((row_f2_opt["recall_risco"] - row_default["recall_risco"]) * 100),
    }


def _plot_threshold_tuning(df_thresh: pd.DataFrame, default_t: float, opt_t: float):
    """Gera visualização do trade-off entre precisão e recall social."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    ax.plot(df_thresh["threshold"], df_thresh["recall_risco"], label="Recall (Alunos em Risco)", color="#e74c3c", linewidth=2.5)
    ax.plot(df_thresh["threshold"], df_thresh["precision_risco"], label="Precisão (Alunos em Risco)", color="#3498db", linewidth=2.0)
    ax.plot(df_thresh["threshold"], df_thresh["f2_risco"], label="F2-Score Risco (Prioriza Recall)", color="#8e44ad", linewidth=2.2, linestyle="-.")
    ax.plot(df_thresh["threshold"], df_thresh["f1_global"], label="F1-Score Global", color="#2ecc71", linewidth=1.8, linestyle="--")
    
    ax.axvline(default_t, color="#7f8c8d", linestyle=":", label=f"Limiar Padrão ({default_t:.2f})")
    ax.axvline(opt_t, color="#e74c3c", linestyle="--", label=f"Limiar Socialmente Ótimo ({opt_t:.2f})")
    
    ax.set_title("Otimização do Limiar de Decisão (Trade-off Precisão vs Recall)", pad=12)
    ax.set_xlabel("Limiar de Decisão (Probabilidade de Corte)")
    ax.set_ylabel("Pontuação da Métrica")
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eval_04_tradeoff_limiar.png", IMAGES_DIR / "eval_04_tradeoff_limiar.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()

