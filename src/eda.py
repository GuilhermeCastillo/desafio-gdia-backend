import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def conferir_missing(df):
    print("Valores nulos e 'unknown':")
    for col in df.columns:
        n_null = df[col].isnull().sum()
        n_unknown = (df[col] == "unknown").sum() if df[col].dtype == "object" else 0
        total = n_null + n_unknown
        if total > 0:
            perc = round(100 * total / len(df), 2)
            print(f"{col}: {total} missing ({perc}%)")
    print("Verificação finalizada.\n")


def categorize_corr(val):
    if abs(val) >= 0.7:
        return "Alto"
    elif abs(val) >= 0.4:
        return "Moderado"
    elif abs(val) >= 0.1:
        return "Baixo"
    else:
        return "Nenhum"


def executar_eda(caminho_csv: str):
    print("Lendo e carregando os dados...")
    df = pd.read_csv(caminho_csv, delimiter=",")

    con = duckdb.connect(database=":memory:")
    con.register("bank", df)

    print("\nQuantidade de linhas e colunas:")
    n_linhas = con.execute("SELECT COUNT(*) FROM bank").fetchone()[0]
    n_colunas = con.execute(
        """
        SELECT COUNT(*) 
        FROM information_schema.columns 
        WHERE table_name = 'bank'
    """
    ).fetchone()[0]
    print(f"O dataset possui {n_linhas} linhas e {n_colunas} colunas.\n")

    conferir_missing(df)

    print("Contando 'unknown' em colunas categóricas:")
    cols = df.select_dtypes(include="object").columns.to_list()
    query = "SELECT\n"
    query += ",\n".join(
        [
            f"COUNT(*) FILTER (WHERE \"{col}\" = 'unknown') AS unknown_{col}"
            for col in cols
        ]
    )
    query += "\nFROM bank;"
    result = con.execute(query).fetchdf()
    print(result.T, "\n")

    print("Correlação entre variáveis numéricas:")
    numerical_cols = df.select_dtypes(include="number").columns.to_list()
    print("Colunas numéricas:", numerical_cols)

    corr_matrix = pd.DataFrame(
        np.eye(len(numerical_cols)), columns=numerical_cols, index=numerical_cols
    )
    results = []

    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            col1 = numerical_cols[i]
            col2 = numerical_cols[j]
            query = f"SELECT CORR({col1}, {col2}) FROM bank"
            corr_value = con.execute(query).fetchone()[0]
            results.append((col1, col2, corr_value))

    for col1, col2, corr in results:
        corr_matrix.loc[col1, col2] = corr
        corr_matrix.loc[col2, col1] = corr

    print("\nMatriz de Correlação (valores numéricos):\n", corr_matrix, "\n")

    annot_labels = corr_matrix.applymap(categorize_corr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=annot_labels, fmt="", cmap="coolwarm", center=0)
    plt.title("Matriz de Correlação (Categorias)")
    plt.tight_layout()
    plt.show()

    con.close()
    return df
