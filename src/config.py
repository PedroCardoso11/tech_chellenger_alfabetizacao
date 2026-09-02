"""
Módulo de Configuração Global (src/config.py)
Centraliza parâmetros globais, caminhos de diretórios, seeds de reproducibilidade,
definições de atributos e configurações de modelagem e otimização.
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
# Definição de Variáveis (Features & Target)
# ==========================================
TARGET_COLUMN = "alfabetizado"  # 1 = Alfabetizado, 0 = Não alfabetizado

# 1. Variáveis Educacionais
NUMERICAL_FEATURES_EDU = [
    "frequencia_escolar",          # Taxa de presença do aluno (0 a 100%)
    "formacao_docente_superior",   # % de professores com ensino superior/pós (0 a 100%)
    "tamanho_turma",               # Número de alunos na turma
    "horas_aula_diarias",          # Horas de permanência diária na escola
]

CATEGORICAL_FEATURES_EDU = [
    "rede",                        # 'Municipal', 'Estadual', 'Privada'
    "infra_agua_filtrada",         # 'Sim', 'Não'
    "infra_biblioteca",            # 'Sim', 'Não'
    "infra_laboratorio_info",      # 'Sim', 'Não'
    "infra_internet_banda_larga",  # 'Sim', 'Não'
    "infra_quadra_esportes",       # 'Sim', 'Não'
]

# 2. Variáveis Territoriais
CATEGORICAL_FEATURES_TERR = [
    "localizacao",                 # 'Urbana', 'Rural'
    "regiao_brasil",               # 'Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'
    "porte_municipio",             # 'Pequeno I', 'Pequeno II', 'Medio', 'Grande', 'Metropole'
    "sigla_uf",                    # Unidade Federativa (27 UFs)
]

NUMERICAL_FEATURES_TERR = [
    "ivs_territorial",             # Índice de Vulnerabilidade Social Municipal (0 a 1)
    "taxa_cobertura_creche_mun",   # Taxa de cobertura de educação infantil municipal (%)
]

# 3. Variáveis Socioeconômicas
NUMERICAL_FEATURES_SOCIO = [
    "renda_per_capita_reais",      # Renda familiar per capita (R$)
    "quantidade_livros_casa",      # Quantidade aproximada de livros no domicílio
]

CATEGORICAL_FEATURES_SOCIO = [
    "beneficiario_bolsa_familia",  # 'Sim', 'Não'
    "escolaridade_mae",            # 'Sem instrucao', 'Fundamental incompleto', 'Fundamental completo', 'Medio completo', 'Superior completo'
    "acesso_internet_casa",        # 'Sim', 'Não'
    "tem_computador_ou_tablet",    # 'Sim', 'Não'
]

# Consolidação de Features Base
BASE_NUMERICAL_FEATURES = (
    NUMERICAL_FEATURES_EDU
    + NUMERICAL_FEATURES_TERR
    + NUMERICAL_FEATURES_SOCIO
)

# Features Numéricas Geradas por Feature Engineering
ENGINEERED_NUMERICAL_FEATURES = [
    "indice_infraestrutura_composto",
    "indice_capital_cultural_casa",
    "indice_vulnerabilidade_familiar",
    "razao_engajamento_turma",
]

ALL_NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

CATEGORICAL_NOMINAL_LOW_CARD = [
    "rede",
    "localizacao",
    "regiao_brasil",
    "infra_agua_filtrada",
    "infra_biblioteca",
    "infra_laboratorio_info",
    "infra_internet_banda_larga",
    "infra_quadra_esportes",
    "beneficiario_bolsa_familia",
    "acesso_internet_casa",
    "tem_computador_ou_tablet",
]

CATEGORICAL_ORDINAL = [
    "porte_municipio",
    "escolaridade_mae",
]

CATEGORICAL_HIGH_CARD = [
    "sigla_uf"
]

ORDINAL_MAPPINGS = {
    "porte_municipio": {
        "Pequeno I": 1,
        "Pequeno II": 2,
        "Medio": 3,
        "Grande": 4,
        "Metropole": 5,
    },
    "escolaridade_mae": {
        "Sem instrucao": 0,
        "Fundamental incompleto": 1,
        "Fundamental completo": 2,
        "Medio completo": 3,
        "Superior completo": 4,
    }
}

