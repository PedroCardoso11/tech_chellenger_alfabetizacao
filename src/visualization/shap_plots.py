"""
Módulo de Explicabilidade de Modelos (src/visualization/shap_plots.py)
Implementa interpretabilidade via SHAP (TreeExplainer) e Feature Importance global,
decompondo o impacto preditivo entre as dimensões Educacional, Territorial e Socioeconômica.
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, IMAGES_DIR, RANDOM_STATE
from src.preprocessing.pipeline import extract_transformed_feature_names


def explain_model_with_shap(
    best_pipeline: Pipeline,
    X_test_raw: pd.DataFrame,
    feature_dict: dict,
    sample_size: int = 1500,
    save_figures: bool = True
) -> Dict[str, Any]:
    """
    Calcula valores SHAP para o modelo em árvore a partir do Pipeline treinado.
    """
    print("\n" + "=" * 70)
    print(" 5. EXPLICABILIDADE DO MODELO (EXPLAINABLE AI - SHAP)")
    print("=" * 70)
    
    np.random.seed(RANDOM_STATE)
    n_samples = min(sample_size, X_test_raw.shape[0])
    idx_sample = np.random.choice(X_test_raw.shape[0], size=n_samples, replace=False)
    X_sample_raw = X_test_raw.iloc[idx_sample].copy()
    
    # Extrair etapas do Pipeline
    fe = best_pipeline.named_steps["feature_engineer"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    
    # Transformar dados para a entrada do estimador em árvore
    X_sample_fe = fe.transform(X_sample_raw)
    X_sample_proc = preprocessor.transform(X_sample_fe)
    
    feature_names = extract_transformed_feature_names(preprocessor, feature_dict)
    df_sample = pd.DataFrame(X_sample_proc, columns=feature_names)
    
    print(f"[SHAP] Calculando valores SHAP via TreeExplainer ({n_samples:,} amostras)...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(df_sample)
    
    # Extrair valores brutos para a classe positiva (1 = Alfabetizado) se for multidimensional
    if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] == 2:
        shap_vals_matrix = shap_values.values[:, :, 1]
    else:
        shap_vals_matrix = shap_values.values
        
    mean_abs_shap = np.mean(np.abs(shap_vals_matrix), axis=0)
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    df_importance["dominio"] = df_importance["feature"].apply(_classify_domain)
    
    print("\n--- Top 10 Variáveis Mais Determinantes (Global Feature Importance SHAP) ---")
    for i, row in df_importance.head(10).iterrows():
        print(f"  {i+1:>2}. {row['feature']:<35} | {row['dominio']:<15} | Impacto Médio: {row['mean_abs_shap']:.4f}")
        
    # Decomposição do Impacto por Domínio
    domain_agg = df_importance.groupby("dominio")["mean_abs_shap"].sum()
    domain_pct = (domain_agg / domain_agg.sum()) * 100
    
    print("\n--- Participação Relativa dos Pilares na Predição da Alfabetização ---")
    for dom, pct in domain_pct.items():
        print(f"  * Pilar {dom:<16}: {pct:>5.1f}% do impacto total explicativo")
        
    if save_figures:
        _plot_shap_figures(explainer, shap_values, df_sample, df_importance, domain_pct)
        
    return {
        "top_features": df_importance.head(15).to_dict(orient="records"),
        "domain_shares": domain_pct.to_dict(),
    }


def _classify_domain(feature_name: str) -> str:
    """Classifica uma feature entre Educacional, Territorial ou Socioeconômico."""
    feat_lower = feature_name.lower()
    if any(k in feat_lower for k in ["freq", "docente", "turma", "hora", "infra", "rede", "escola", "engajamento"]):
        return "Educacional"
    elif any(k in feat_lower for k in ["uf", "regiao", "porte", "localizacao", "ivs_terr", "creche", "rural", "urbana"]):
        return "Territorial"
    elif any(k in feat_lower for k in ["renda", "bolsa", "familia", "mae", "livro", "comp", "internet", "vuln", "cultural"]):
        return "Socioeconômico"
    else:
        return "Educacional"


def _plot_shap_figures(
    explainer: Any,
    shap_values: Any,
    df_sample: pd.DataFrame,
    df_importance: pd.DataFrame,
    domain_pct: pd.Series
):
    """Gera gráficos de interpretabilidade SHAP e salva em reports/figures e images."""
    for d in [FIGURES_DIR, IMAGES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # FIGURA 1: Beeswarm Summary Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, df_sample, max_display=15, show=False)
    plt.title("SHAP Beeswarm: Impacto e Direção das Features na Alfabetização", pad=15, fontsize=13, fontweight="bold")
    plt.tight_layout()
    for dest in [FIGURES_DIR / "shap_01_beeswarm_summary.png", IMAGES_DIR / "shap_01_beeswarm_summary.png"]:
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 2: Bar Feature Importance Global
    fig, ax = plt.subplots(figsize=(10, 6))
    top_15 = df_importance.head(15).sort_values(by="mean_abs_shap", ascending=True)
    palette_colors = {"Educacional": "#2980b9", "Socioeconômico": "#e67e22", "Territorial": "#27ae60"}
    bar_colors = [palette_colors.get(d, "#7f8c8d") for d in top_15["dominio"]]
    
    ax.barh(top_15["feature"], top_15["mean_abs_shap"], color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Importância Média (|SHAP Value|)")
    ax.set_title("Ranking das 15 Variáveis Mais Relevantes para a Predição (SHAP)", pad=12)
    
    handles = [plt.Rectangle((0,0),1,1, color=color, label=label) for label, color in palette_colors.items()]
    ax.legend(handles=handles, title="Dimensão", loc="lower right", frameon=True)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "shap_02_global_importance.png", IMAGES_DIR / "shap_02_global_importance.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()

    # FIGURA 3: Participação Relativa por Domínio (Donut Chart)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_donut = [palette_colors.get(d, "#95a5a6") for d in domain_pct.index]
    wedges, texts, autotexts = ax.pie(
        domain_pct.values,
        labels=domain_pct.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_donut,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11, fontweight="bold")
    )
    ax.set_title("Participação das Dimensões na Capacidade Preditiva do Modelo", pad=15)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "shap_03_distribuicao_pilares.png", IMAGES_DIR / "shap_03_distribuicao_pilares.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()

    # FIGURA 4: Waterfall Plot de Caso Individual
    try:
        if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] == 2:
            single_explanation = shap.Explanation(
                values=shap_values.values[0, :, 1],
                base_values=shap_values.base_values[0, 1] if hasattr(shap_values.base_values, "__len__") else explainer.expected_value[1],
                data=df_sample.iloc[0].values,
                feature_names=list(df_sample.columns)
            )
        else:
            single_explanation = shap.Explanation(
                values=shap_values.values[0],
                base_values=shap_values.base_values[0] if hasattr(shap_values.base_values, "__len__") else explainer.expected_value,
                data=df_sample.iloc[0].values,
                feature_names=list(df_sample.columns)
            )
            
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(single_explanation, max_display=10, show=False)
        plt.title("Diagnóstico Individualizado SHAP (Aluno com Fatores de Risco)", pad=15, fontsize=12, fontweight="bold")
        plt.tight_layout()
        for dest in [FIGURES_DIR / "shap_04_waterfall_caso_individual.png", IMAGES_DIR / "shap_04_waterfall_caso_individual.png"]:
            plt.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[SHAP] Aviso na geração do waterfall plot: {e}")
        
    print(f"[SHAP] Gráficos de interpretabilidade salvos em: {FIGURES_DIR} e {IMAGES_DIR}")

