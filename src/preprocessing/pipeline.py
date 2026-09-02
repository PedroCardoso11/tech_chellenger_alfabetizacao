"""
Módulo de Engenharia de Atributos e Pipeline de Pré-processamento (src/preprocessing/pipeline.py)
Criação de features compostas educacionais, imputação, escalonamento e codificação
categórica com garantia estrita de Zero Data Leakage.
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    BASE_NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    CATEGORICAL_NOMINAL_LOW_CARD,
    CATEGORICAL_ORDINAL,
    CATEGORICAL_HIGH_CARD,
)


class EducationFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado Scikit-Learn para criação de índices compostos
    e atributos educacionais / socioeconômicos derivados de domínio.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Índice Composto de Infraestrutura Escolar (0 a 1)
        pesos_infra = {
            "infra_agua_filtrada": 0.15,
            "infra_biblioteca": 0.30,
            "infra_laboratorio_info": 0.20,
            "infra_internet_banda_larga": 0.25,
            "infra_quadra_esportes": 0.10,
        }
        score_infra = np.zeros(len(X_out))
        for col, peso in pesos_infra.items():
            if col in X_out.columns:
                val = (X_out[col] == "Sim").astype(float)
                score_infra += val * peso
        X_out["indice_infraestrutura_composto"] = np.round(score_infra, 3)

        # 2. Índice de Capital Cultural e Tecnológico Domiciliar (0 a 1)
        livros_norm = np.clip(X_out.get("quantidade_livros_casa", 0) / 30.0, 0, 1)
        tem_comp = (X_out.get("tem_computador_ou_tablet", "Não") == "Sim").astype(float)
        tem_net = (X_out.get("acesso_internet_casa", "Não") == "Sim").astype(float)
        
        esc_map = {
            "Sem instrucao": 0.0,
            "Fundamental incompleto": 0.25,
            "Fundamental completo": 0.50,
            "Medio completo": 0.75,
            "Superior completo": 1.0,
        }
        esc_val = X_out.get("escolaridade_mae", "Sem instrucao").map(esc_map).fillna(0.3)
        X_out["indice_capital_cultural_casa"] = np.round(
            0.35 * esc_val + 0.25 * livros_norm + 0.20 * tem_comp + 0.20 * tem_net, 3
        )

        # 3. Índice de Vulnerabilidade Familiar (0 a 1)
        renda = X_out.get("renda_per_capita_reais", 800.0).fillna(800.0)
        renda_vuln = np.clip(1.0 - (renda / 2500.0), 0.0, 1.0)
        bf_vuln = (X_out.get("beneficiario_bolsa_familia", "Não") == "Sim").astype(float)
        ivs_terr = X_out.get("ivs_territorial", 0.3).fillna(0.3)
        X_out["indice_vulnerabilidade_familiar"] = np.round(
            0.40 * renda_vuln + 0.35 * bf_vuln + 0.25 * ivs_terr, 3
        )

        # 4. Razão de Engajamento e Atenção Individual (Frequência / Tamanho Turma)
        freq = X_out.get("frequencia_escolar", 80.0).fillna(80.0)
        turma = X_out.get("tamanho_turma", 25.0).replace(0, 25.0).fillna(25.0)
        X_out["razao_engajamento_turma"] = np.round(freq / turma, 2)

        return X_out


def build_feature_dictionary(df_cols: List[str]) -> Dict[str, List[str]]:
    """Gera o dicionário com os grupos de atributos."""
    return {
        "numeric": BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES,
        "nominal": [c for c in CATEGORICAL_NOMINAL_LOW_CARD if c in df_cols],
        "ordinal": [c for c in CATEGORICAL_ORDINAL if c in df_cols],
        "high_card": [c for c in CATEGORICAL_HIGH_CARD if c in df_cols],
    }


def build_preprocessor_pipeline(feature_dict: dict) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer do Scikit-Learn contendo imputação,
    escalonamento numérico e codificação categórica.
    """
    numeric_cols = feature_dict["numeric"]
    nominal_cols = feature_dict["nominal"]
    ordinal_cols = feature_dict["ordinal"]
    high_card_cols = feature_dict["high_card"]

    # Pipeline Numérico: Imputação por Mediana + RobustScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    # Pipeline Categórico Nominal: Imputação por Moda + OneHotEncoder
    nom_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    # Pipeline Categórico Ordinal: Imputação por Moda + OrdinalEncoder
    ordinal_categories = [
        ["Pequeno I", "Pequeno II", "Medio", "Grande", "Metropole"],
        ["Sem instrucao", "Fundamental incompleto", "Fundamental completo", "Medio completo", "Superior completo"]
    ]
    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", RobustScaler()),
    ])
    
    # Pipeline para Alta Cardinalidade (UF): OneHotEncoder
    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("nom", nom_pipeline, nominal_cols),
            ("ord", ord_pipeline, ordinal_cols),
            ("high_card", high_card_pipeline, high_card_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def extract_transformed_feature_names(fitted_preprocessor: ColumnTransformer, feature_dict: dict) -> List[str]:
    """Extrai os nomes finais de todas as colunas pós-transformação."""
    feature_names = []
    
    # Numéricas
    feature_names.extend(feature_dict["numeric"])
    
    # Nominais One-Hot
    try:
        nom_ohe = fitted_preprocessor.named_transformers_["nom"].named_steps["onehot"]
        nom_cols = nom_ohe.get_feature_names_out(feature_dict["nominal"])
        feature_names.extend(list(nom_cols))
    except Exception:
        pass
        
    # Ordinais
    feature_names.extend(feature_dict["ordinal"])
    
    # High Card (UF) One-Hot
    try:
        uf_ohe = fitted_preprocessor.named_transformers_["high_card"].named_steps["onehot"]
        uf_cols = uf_ohe.get_feature_names_out(feature_dict["high_card"])
        feature_names.extend(list(uf_cols))
    except Exception:
        pass
        
    return feature_names

