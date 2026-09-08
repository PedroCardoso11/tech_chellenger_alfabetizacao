"""
Módulo de Configuração Global (src/config.py)
Centraliza parâmetros globais, caminhos de diretórios, seeds de reproducibilidade,
definições de atributos 100% REAIS e configurações de modelagem e otimização.
"""

import os
from pathlib import Path

# ==========================================
# Diretórios e Caminhos do Projeto
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
IMAGES_DIR = ROOT_DIR / "images"
MODELS_DIR = ROOT_DIR / "models_saved"

# Criação automática de diretórios necessários
for directory in [DATA_DIR, REPORTS_DIR, FIGURES_DIR, IMAGES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# Parâmetros de Reproducibilidade e Divisão
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5
N_OPTUNA_TRIALS = 25

# ==========================================
# Definição de Variáveis 100% Reais (Features & Target)
# ==========================================
TARGET_COLUMN = "alfabetizado"  # 1 = Alfabetizado, 0 = Não alfabetizado

# 1. Variáveis Educacionais Reais (Escola e Aluno)
NUMERICAL_FEATURES_EDU = [
    "peso_aluno",                         # Peso amostral estatístico oficial do aluno no SAEB
    "escola_total_alunos",                # Porte da escola (alunos avaliados no 2º ano)
    "escola_percentual_presenca",         # Taxa de presença real dos alunos da escola na avaliação
    "escola_percentual_nao_alfabetizado", # Taxa histórica agregada de não alfabetização da escola
    "escola_ranking_municipio",           # Posição da escola no ranking do município
    "escola_ranking_uf",                  # Posição da escola no ranking do estado
]

CATEGORICAL_FEATURES_EDU = [
    "rede",                               # 'Municipal', 'Estadual', 'Privada', 'Federal'
]

# 2. Variáveis Socioeconômicas Reais (Bolsa Família Municipal)
NUMERICAL_FEATURES_SOCIO = [
    "total_beneficiarios",                # Total de famílias beneficiárias do Bolsa Família no município
    "valor_total_pago",                   # Volume financeiro total repassado ao município
    "bf_beneficio_medio",                 # Valor médio do benefício por família no município (R$)
]

CATEGORICAL_FEATURES_SOCIO = []

# 3. Variáveis Territoriais Reais (Município e UF)
NUMERICAL_FEATURES_TERR = [
    "mun_meta_alfabetizacao",             # Meta anual pactuada do CNCA para o município
    "mun_distancia_meta",                 # Distância entre o resultado e a meta pactuada
    "uf_desvio_padrao_resultado",         # Dispersão/desigualdade de aprendizado entre municípios do estado
    "uf_amplitude_resultado",             # Amplitude de desempenho da UF (diferença máx - mín)
    "uf_percentual_municipios_abaixo_meta", # % de municípios da UF que não atingiram a meta
]

CATEGORICAL_FEATURES_TERR = [
    "regiao_brasil",                      # 'Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'
    "mun_classe_risco",                   # 'Meta atingida', 'Ate 5 p.p. abaixo', 'Entre 5 e 10 p.p. abaixo', 'Mais de 10 p.p. abaixo'
    "sigla_uf",                           # Unidade Federativa (27 UFs)
]

# Consolidação de Features Base Reais
BASE_NUMERICAL_FEATURES = (
    NUMERICAL_FEATURES_EDU
    + NUMERICAL_FEATURES_SOCIO
    + NUMERICAL_FEATURES_TERR
)

# Features Numéricas Derivadas (Domain Feature Engineering Real)
ENGINEERED_NUMERICAL_FEATURES = [
    "razao_desempenho_escola_uf",         # Desempenho relativo da escola em relação ao estado
    "indice_engajamento_escola",          # Interação entre presença escolar e proficiência agregada
    "razao_beneficiarios_porte_escola",   # Pressão de vulnerabilidade social no entorno da escola
]

ALL_NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

CATEGORICAL_NOMINAL_LOW_CARD = [
    "rede",
    "regiao_brasil",
]

CATEGORICAL_ORDINAL = [
    "mun_classe_risco",
]

CATEGORICAL_HIGH_CARD = [
    "sigla_uf",
]

ORDINAL_MAPPINGS = {
    "mun_classe_risco": {
        "Meta atingida": 3,
        "Ate 5 p.p. abaixo": 2,
        "Entre 5 e 10 p.p. abaixo": 1,
        "Mais de 10 p.p. abaixo": 0,
    }
}
