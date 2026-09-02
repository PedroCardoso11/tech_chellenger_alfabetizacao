"""
Módulo de Avaliação de Desempenho (src/evaluation/metrics.py)
Cálculo de métricas completas no conjunto de teste independente (Holdout 20%),
geração de gráficos diagnósticos (ROC, PR, Matriz de Confusão) e persistência de relatórios.
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)

from src.config import FIGURES_DIR, IMAGES_DIR, REPORTS_DIR


def evaluate_all_models_on_test(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    save_figures: bool = True
) -> pd.DataFrame:
    """
    Avalia múltiplos pipelines treinados no conjunto de teste independente.
    
    Returns:
        DataFrame consolidado de métricas em teste.
    """
    print("\n" + "=" * 70)
    print(" 4. AVALIAÇÃO DE DESEMPENHO NO CONJUNTO DE TESTE INDEPENDENTE")
    print("=" * 70)
    
    metrics_records = []
    roc_data = {}
    pr_data = {}
    cm_data = {}
    
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        
        # Probabilidades para a classe 1 (Alfabetizado)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        loss = log_loss(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        # Recall da classe vulnerável (0 = Não Alfabetizado)
        rec_nao_alfab = recall_score(1 - y_test, 1 - y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        cm_data[name] = cm
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc)
        
        precs, recs, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (recs, precs, pr_auc)
        
        record = {
            "Modelo": name,
            "Acurácia": round(acc, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "F1-Score": round(f1, 4),
            "Recall (Alfabetizado)": round(rec, 4),
            "Recall (Não Alfab - Crítico)": round(rec_nao_alfab, 4),
            "Precisão": round(prec, 4),
            "Log-Loss": round(loss, 4),
            "Brier Score": round(brier, 4),
        }
        metrics_records.append(record)
        
    df_metrics = pd.DataFrame(metrics_records).sort_values(by="ROC-AUC", ascending=False)
    
    print("\n--- Tabela Comparativa no Conjunto de Teste (20% Holdout) ---")
    print(df_metrics.to_string(index=False))
    
    # Salvar métricas em JSON
    metrics_json_path = REPORTS_DIR / "metrics_summary.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_records, f, indent=4, ensure_ascii=False)
    print(f"\n[EVALUATION] Sumário quantitativo salvo em: {metrics_json_path}")
    
    if save_figures:
        _plot_confusion_matrices(cm_data)
        _plot_roc_curves(roc_data)
        _plot_pr_curves(pr_data)
        
    return df_metrics


def _plot_confusion_matrices(cm_dict: Dict[str, np.ndarray]):
    """Gera painel com matrizes de confusão normalizadas para todos os modelos."""
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
        
    labels = ["Não Alfab (0)", "Alfab (1)"]
    
    for ax, (name, cm) in zip(axes, cm_dict.items()):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        annot_text = np.array([
            [f"{val:,}\n({pct:.1%})" for val, pct in zip(row_val, row_pct)]
            for row_val, row_pct in zip(cm, cm_norm)
        ])
        
        sns.heatmap(cm_norm, annot=annot_text, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 10, "fontweight": "bold"})
        ax.set_title(f"{name.replace('_', ' ')}", pad=10)
        ax.set_xlabel("Predição do Modelo")
        ax.set_ylabel("Realidade do Aluno")
        
    plt.suptitle("Matrizes de Confusão Normalizadas (Conjunto de Teste)", fontsize=14, y=1.03)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eval_01_matrizes_confusao.png", IMAGES_DIR / "eval_01_matrizes_confusao.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()


def _plot_roc_curves(roc_dict: Dict[str, tuple]):
    """Gera curva ROC comparativa de todos os modelos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#7f8c8d", "#2980b9", "#e67e22", "#27ae60", "#8e44ad"]
    
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_dict.items()):
        color = colors[i % len(colors)]
        linestyle = "--" if "Baseline" in name else "-"
        linewidth = 2.5 if ("LightGBM" in name or "XGBoost" in name) else 1.8
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth,
                label=f"{name.replace('_', ' ')} (AUC = {roc_auc:.4f})")
        
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", label="Classificador Aleatório (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade / Recall)")
    ax.set_title("Curvas ROC (Receiver Operating Characteristic) Comparativas", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eval_02_curvas_roc.png", IMAGES_DIR / "eval_02_curvas_roc.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()


def _plot_pr_curves(pr_dict: Dict[str, tuple]):
    """Gera curvas Precision-Recall comparativas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#7f8c8d", "#2980b9", "#e67e22", "#27ae60", "#8e44ad"]
    
    for i, (name, (recall, precision, pr_auc)) in enumerate(pr_dict.items()):
        color = colors[i % len(colors)]
        linestyle = "--" if "Baseline" in name else "-"
        linewidth = 2.5 if ("LightGBM" in name or "XGBoost" in name) else 1.8
        ax.plot(recall, precision, color=color, linestyle=linestyle, linewidth=linewidth,
                label=f"{name.replace('_', ' ')} (PR-AUC = {pr_auc:.4f})")
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Revocação (Recall)")
    ax.set_ylabel("Precisão (Precision)")
    ax.set_title("Curvas Precision-Recall (PR) Comparativas", pad=12)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eval_03_curvas_pr.png", IMAGES_DIR / "eval_03_curvas_pr.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()

