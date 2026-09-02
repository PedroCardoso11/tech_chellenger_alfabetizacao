"""
Módulo de Análise Exploratória e Visualizações de Dados (src/visualization/eda_plots.py)
Geração de diagnóstico estatístico, detecção de outliers via IQR e plots analíticos.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, IMAGES_DIR, TARGET_COLUMN

# Configuração estética padronizada
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})


def perform_eda(df: pd.DataFrame, save_figures: bool = True) -> dict:
    """
    Executa a Análise Exploratória de Dados completa e gera gráficos estatísticos.
    """
    print("=" * 70)
    print(" 1. ANÁLISE EXPLORATÓRIA DOS DADOS (EDA) & DIAGNÓSTICO ESTATÍSTICO")
    print("=" * 70)
    
    # 1.1 Informações Estruturais
    print(f"\n[EDA] Dimensões do Dataset: {df.shape[0]:,} linhas e {df.shape[1]} colunas.")
    
    # 1.2 Análise de Valores Ausentes (Missing Values)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        "Total Ausentes": missing_counts,
        "Percentual (%)": missing_pct.round(2)
    }).query("`Total Ausentes` > 0").sort_values(by="Percentual (%)", ascending=False)
    
    print("\n--- Diagnóstico de Valores Ausentes ---")
    if not missing_df.empty:
        print(missing_df.to_string())
    else:
        print("Nenhum valor ausente detectado.")
        
    # 1.3 Análise da Variável Alvo e Desbalanceamento
    target_counts = df[TARGET_COLUMN].value_counts()
    target_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    class_ratio = target_counts.max() / target_counts.min() if target_counts.min() > 0 else np.nan
    
    print("\n--- Distribuição da Variável Alvo (Alfabetização) ---")
    print(f"Classe 1 (Alfabetizado):     {target_counts.get(1, 0):>6,} ({target_pct.get(1, 0):.2f}%)")
    print(f"Classe 0 (Não Alfabetizado): {target_counts.get(0, 0):>6,} ({target_pct.get(0, 0):.2f}%)")
    print(f"Razão de Desbalanceamento:  {class_ratio:.2f}:1")
    
    # 1.4 Identificação de Outliers em Variáveis Numéricas (Método IQR)
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COLUMN, errors="ignore")
    outliers_summary = {}
    
    print("\n--- Detecção de Outliers via Intervalo Interquartil (IQR) ---")
    for col in num_cols:
        series = df[col].dropna()
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        n_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        pct_outliers = (n_outliers / len(series)) * 100
        outliers_summary[col] = {
            "n_outliers": n_outliers,
            "pct_outliers": round(pct_outliers, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        }
        if n_outliers > 0:
            print(f"  - {col:<26}: {n_outliers:>5,} outliers ({pct_outliers:.2f}%) | Limites: [{lower_bound:.1f}, {upper_bound:.1f}]")

    # 1.5 Geração de Gráficos Analíticos
    if save_figures:
        _generate_eda_plots(df, target_counts, target_pct, num_cols)
        
    return {
        "missing_summary": missing_df.to_dict(),
        "target_distribution": target_pct.to_dict(),
        "outliers_summary": outliers_summary
    }


def _generate_eda_plots(df: pd.DataFrame, target_counts: pd.Series, target_pct: pd.Series, num_cols: pd.Index):
    """Gera e salva figuras analíticas da fase de EDA em reports/figures e images."""
    for d in [FIGURES_DIR, IMAGES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # FIGURA 1: Distribuição da Variável Alvo
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Não Alfabetizado (0)", "Alfabetizado (1)"]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, [target_counts.get(0, 0), target_counts.get(1, 0)], color=colors, edgecolor="black", alpha=0.85, width=0.5)
    
    for bar, pct in zip(bars, [target_pct.get(0, 0), target_pct.get(1, 0)]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + (target_counts.max() * 0.02),
                f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_title("Distribuição da Variável Alvo: Alfabetização no 2º Ano EF", pad=15)
    ax.set_ylabel("Total de Alunos")
    ax.set_ylim(0, target_counts.max() * 1.18)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eda_01_distribuicao_alvo.png", IMAGES_DIR / "eda_01_distribuicao_alvo.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()

    # FIGURA 2: Impacto Socioeconômico e Frequência
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df_plot = df.copy()
    df_plot["Status Alfabetização"] = df_plot[TARGET_COLUMN].map({1: "Alfabetizado", 0: "Não Alfabetizado"})
    palette = {"Alfabetizado": "#2ecc71", "Não Alfabetizado": "#e74c3c"}
    
    sns.boxplot(data=df_plot, x="Status Alfabetização", y="frequencia_escolar", hue="Status Alfabetização", ax=axes[0], palette=palette, legend=False, boxprops=dict(alpha=0.8))
    axes[0].set_title("Frequência Escolar (%)")
    axes[0].set_ylabel("Presença (%)")
    axes[0].set_xlabel("")
    
    sns.boxplot(data=df_plot[df_plot["renda_per_capita_reais"] < 3500], x="Status Alfabetização", y="renda_per_capita_reais", hue="Status Alfabetização", ax=axes[1], palette=palette, legend=False, boxprops=dict(alpha=0.8))
    axes[1].set_title("Renda Familiar Per Capita (R$)")
    axes[1].set_ylabel("Renda (R$)")
    axes[1].set_xlabel("")
    
    sns.boxplot(data=df_plot, x="Status Alfabetização", y="ivs_territorial", hue="Status Alfabetização", ax=axes[2], palette=palette, legend=False, boxprops=dict(alpha=0.8))
    axes[2].set_title("Índice de Vulnerabilidade Social (IVS)")
    axes[2].set_ylabel("IVS Municipal")
    axes[2].set_xlabel("")
    
    plt.suptitle("Determinantes Educacionais e Socioeconômicos da Alfabetização", fontsize=15, y=1.02)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eda_02_fatores_socioeconomicos.png", IMAGES_DIR / "eda_02_fatores_socioeconomicos.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()

    # FIGURA 3: Taxa de Alfabetização por Região e Rede
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    reg_rate = df.groupby("regiao_brasil")[TARGET_COLUMN].mean().sort_values(ascending=False) * 100
    sns.barplot(x=reg_rate.index, y=reg_rate.values, hue=reg_rate.index, ax=axes[0], palette="Blues_r", legend=False, edgecolor="black", alpha=0.85)
    axes[0].set_title("Taxa de Alfabetização por Grande Região (%)")
    axes[0].set_ylabel("Taxa (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(reg_rate.values):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
        
    rede_rate = df.groupby("rede")[TARGET_COLUMN].mean().sort_values(ascending=False) * 100
    sns.barplot(x=rede_rate.index, y=rede_rate.values, hue=rede_rate.index, ax=axes[1], palette="Purples_r", legend=False, edgecolor="black", alpha=0.85)
    axes[1].set_title("Taxa de Alfabetização por Dependência Administrativa (%)")
    axes[1].set_ylabel("Taxa (%)")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(rede_rate.values):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
        
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eda_03_territorio_e_rede.png", IMAGES_DIR / "eda_03_territorio_e_rede.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()

    # FIGURA 4: Matriz de Correlação Numérica
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[list(num_cols) + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Matriz de Correlação Linear (Pearson) das Variáveis Numéricas", pad=12)
    plt.tight_layout()
    for dest in [FIGURES_DIR / "eda_04_matriz_correlacao.png", IMAGES_DIR / "eda_04_matriz_correlacao.png"]:
        plt.savefig(dest, dpi=300)
    plt.close()
    
    print(f"[EDA] 4 figuras analíticas salvas em: {FIGURES_DIR} e {IMAGES_DIR}")

