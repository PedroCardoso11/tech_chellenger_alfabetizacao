"""
Módulo de Carregamento e Integração de Dados 100% REAIS (src/preprocessing/data_loader.py)
Responsável por carregar dados do Data Lakehouse (camadas Silver e Gold),
integrando a tabela fato de alunos (2,12M de registros) com dimensões escolares,
municipais, socioeconômicas (Bolsa Família) e indicadores territoriais de metas.
ZERO dados sintéticos ou simulações estatísticas.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, ROOT_DIR, DATA_DIR


def _find_table_path(relative_subpath: str) -> Optional[Path]:
    """
    Busca um arquivo parquet em DATA_DIR ou ROOT_DIR de forma dinâmica.
    """
    candidate1 = DATA_DIR / relative_subpath
    if candidate1.exists():
        return candidate1
    
    candidate2 = ROOT_DIR / relative_subpath
    if candidate2.exists():
        return candidate2
    
    target_filename = Path(relative_subpath).name
    matches = list(DATA_DIR.glob(f"**/{target_filename}"))
    if matches:
        matches_sorted = sorted(matches, key=lambda p: str(p), reverse=True)
        return matches_sorted[0]
        
    return None


def load_gold_silver_data(sample_size: int = 30000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Carrega e consolida dados 100% REAIS das tabelas Silver e Gold do Data Lakehouse,
    integrando microdados de alunos com indicadores de escola, município e estado.
    
    Args:
        sample_size: Tamanho da amostra estratificada para processamento de ML.
        seed: Semente aleatória para reproducibilidade.
        
    Returns:
        pd.DataFrame com dados 100% reais integrados.
    """
    np.random.seed(seed)
    
    # 1. Carregar tabela fato de alunos (Silver)
    fato_aluno_path = _find_table_path(
        "silver/fato_aluno_alfabetizacao/execution_date=2026-08-31/ano=2024/fato_aluno_alfabetizacao.parquet"
    ) or _find_table_path("silver/fato_aluno_alfabetizacao/fato_aluno_alfabetizacao.parquet")
    
    if not fato_aluno_path or not fato_aluno_path.exists():
        raise FileNotFoundError(f"[DATA LOADER ERRO] Arquivo Silver de alunos não encontrado em: {DATA_DIR}")
        
    print(f"[DATA LOADER] Lendo microdados reais de alunos da camada Silver: {fato_aluno_path}")
    
    colunas_leitura_aluno = [
        "id_aluno", "id_escola", "id_municipio", "rede",
        "preenchimento_caderno", "alfabetizado", "peso_aluno",
        "flag_alfabetizado_preenchido"
    ]
    df_alunos_raw = pd.read_parquet(fato_aluno_path, columns=colunas_leitura_aluno)
    
    # Filtrar alunos com status preenchido e prova realizada (avaliação preenchida)
    df_valid = df_alunos_raw[
        (df_alunos_raw["flag_alfabetizado_preenchido"] == True) &
        (df_alunos_raw["alfabetizado"].notna()) &
        (df_alunos_raw["alfabetizado"].isin(["Sim", "Não", "No", "Nao"])) &
        (df_alunos_raw["preenchimento_caderno"] == "Prova preenchida")
    ].copy()
    
    # Padronizar target alfabetizado (1 = Sim, 0 = Não)
    df_valid["alfabetizado"] = df_valid["alfabetizado"].apply(
        lambda x: 1 if str(x).strip().lower() in ["sim", "1", "s"] else 0
    )
    
    # Amostragem estratificada representativa
    if len(df_valid) > sample_size:
        df_sampled, _ = train_test_split(
            df_valid,
            train_size=sample_size,
            stratify=df_valid["alfabetizado"],
            random_state=seed
        )
        df_sampled = df_sampled.reset_index(drop=True)
    else:
        df_sampled = df_valid.reset_index(drop=True)
        
    print(f"[DATA LOADER] Amostra estratificada consolidada de alunos: {df_sampled.shape[0]:,} registros.")

    # 2. Carregar Dimensões Básicas (dim_escola e UFs)
    dim_escola_path = _find_table_path("silver/dim_escola/execution_date=2026-08-31/dim_escola.parquet")
    if dim_escola_path and dim_escola_path.exists():
        df_escola = pd.read_parquet(dim_escola_path)
        df_sampled = df_sampled.merge(
            df_escola[["id_escola", "id_municipio_nome"]].drop_duplicates("id_escola"),
            on="id_escola",
            how="left"
        )
    
    # Mapeamento federativo oficial IBGE
    ibge_to_uf = {
        "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
        "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE", "29": "BA",
        "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
        "41": "PR", "42": "SC", "43": "RS",
        "50": "MS", "51": "MT", "52": "GO", "53": "DF"
    }
    df_sampled["sigla_uf"] = df_sampled["id_municipio"].astype(str).str[:2].map(ibge_to_uf).fillna("SP")
    
    uf_to_region = {
        "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
        "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
        "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
        "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
        "PR": "Sul", "RS": "Sul", "SC": "Sul"
    }
    df_sampled["regiao_brasil"] = df_sampled["sigla_uf"].map(uf_to_region).fillna("Sudeste")

    # 3. Cruzamento com Fato Bolsa Família Municipal (Silver)
    fato_bf_path = _find_table_path("silver/fato_bolsa_familia_municipio/execution_date=2026-08-31/ano=2024/fato_bolsa_familia_municipio.parquet")
    if fato_bf_path and fato_bf_path.exists():
        df_bf = pd.read_parquet(fato_bf_path)
        df_bf["bf_beneficio_medio"] = (df_bf["valor_total_pago"] / df_bf["total_beneficiarios"]).astype(float).round(2)
        df_sampled = df_sampled.merge(
            df_bf[["id_municipio", "total_beneficiarios", "valor_total_pago", "bf_beneficio_medio"]].drop_duplicates("id_municipio"),
            on="id_municipio",
            how="left"
        )

    # 4. Cruzamento com Ranking e Desempenho Real de Escolas (Gold)
    ranking_esc_path = _find_table_path("gold/ranking_escolas_prioritarias/execution_date=2026-08-31/ano=2024/ranking_escolas_prioritarias.parquet")
    if ranking_esc_path and ranking_esc_path.exists():
        df_esc = pd.read_parquet(ranking_esc_path)
        df_esc = df_esc.rename(columns={
            "total_alunos": "escola_total_alunos",
            "percentual_presenca": "escola_percentual_presenca",
            "percentual_nao_alfabetizado": "escola_percentual_nao_alfabetizado",
            "ranking_municipio": "escola_ranking_municipio",
            "ranking_uf": "escola_ranking_uf"
        })
        cols_esc = ["id_escola", "escola_total_alunos", "escola_percentual_presenca", "escola_percentual_nao_alfabetizado", "escola_ranking_municipio", "escola_ranking_uf"]
        df_sampled = df_sampled.merge(
            df_esc[cols_esc].drop_duplicates("id_escola"),
            on="id_escola",
            how="left"
        )

    # 5. Cruzamento com Risco Territorial e Metas do Município (Gold)
    mapa_calor_path = _find_table_path("gold/mapa_calor_territorial/execution_date=2026-08-31/ano=2024/mapa_calor_territorial.parquet")
    if mapa_calor_path and mapa_calor_path.exists():
        df_mapa = pd.read_parquet(mapa_calor_path)
        df_mapa = df_mapa.rename(columns={
            "classe_risco": "mun_classe_risco",
            "meta_alfabetizacao": "mun_meta_alfabetizacao",
            "distancia_meta": "mun_distancia_meta"
        })
        cols_mapa = ["id_municipio", "mun_classe_risco", "mun_meta_alfabetizacao", "mun_distancia_meta"]
        df_sampled = df_sampled.merge(
            df_mapa[cols_mapa].drop_duplicates("id_municipio"),
            on="id_municipio",
            how="left"
        )

    # 6. Cruzamento com Desigualdade Territorial e Metas por UF (Gold)
    desig_uf_path = _find_table_path("gold/desigualdade_territorial_uf/execution_date=2026-08-31/ano=2024/desigualdade_territorial_uf.parquet")
    if desig_uf_path and desig_uf_path.exists():
        df_desig = pd.read_parquet(desig_uf_path)
        df_desig = df_desig.rename(columns={
            "desvio_padrao_resultado": "uf_desvio_padrao_resultado",
            "amplitude_resultado": "uf_amplitude_resultado",
            "percentual_municipios_abaixo_meta": "uf_percentual_municipios_abaixo_meta"
        })
        cols_desig = ["sigla_uf", "uf_desvio_padrao_resultado", "uf_amplitude_resultado", "uf_percentual_municipios_abaixo_meta"]
        df_sampled = df_sampled.merge(
            df_desig[cols_desig].drop_duplicates("sigla_uf"),
            on="sigla_uf",
            how="left"
        )

    # Converter colunas para tipos numéricos/categóricos padrão do pandas
    numeric_cols = [
        "peso_aluno", "total_beneficiarios", "valor_total_pago", "bf_beneficio_medio",
        "escola_total_alunos", "escola_percentual_presenca", "escola_percentual_nao_alfabetizado",
        "escola_ranking_municipio", "escola_ranking_uf", "mun_meta_alfabetizacao",
        "mun_distancia_meta", "uf_desvio_padrao_resultado", "uf_amplitude_resultado",
        "uf_percentual_municipios_abaixo_meta"
    ]
    for col in numeric_cols:
        if col in df_sampled.columns:
            df_sampled[col] = pd.to_numeric(df_sampled[col], errors="coerce")

    string_cols = ["rede", "sigla_uf", "regiao_brasil", "mun_classe_risco"]
    for col in string_cols:
        if col in df_sampled.columns:
            df_sampled[col] = df_sampled[col].astype(str).replace({"<NA>": np.nan, "nan": np.nan, "None": np.nan})

    # Seleção estrita das variáveis finais 100% REAIS (excluindo IDs e proficiência)
    colunas_finais = [
        "alfabetizado",
        "rede",
        "sigla_uf",
        "regiao_brasil",
        "peso_aluno",
        "escola_total_alunos",
        "escola_percentual_presenca",
        "escola_percentual_nao_alfabetizado",
        "escola_ranking_municipio",
        "escola_ranking_uf",
        "total_beneficiarios",
        "valor_total_pago",
        "bf_beneficio_medio",
        "mun_classe_risco",
        "mun_meta_alfabetizacao",
        "mun_distancia_meta",
        "uf_desvio_padrao_resultado",
        "uf_amplitude_resultado",
        "uf_percentual_municipios_abaixo_meta",
    ]
    
    df_final = df_sampled[colunas_finais].copy()
    print(f"[DATA LOADER] Dataset consolidado com 100% DADOS REAIS: {df_final.shape[0]:,} linhas e {df_final.shape[1]} colunas.")
    return df_final
