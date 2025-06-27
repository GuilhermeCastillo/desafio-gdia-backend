import pandas as pd
from sklearn.preprocessing import StandardScaler


def carregar_dados(caminho):
    return pd.read_csv(caminho, delimiter=",")


def tratar_valores_nulos(df):
    df["job"] = df["job"].replace("unknown", df["job"].mode()[0])
    df["education"] = df["education"].replace("unknown", df["education"].mode()[0])
    df["poutcome"] = df["poutcome"].replace("unknown", "sem_contato_anterior")
    return df


def encode_variaveis(df):
    binarias = ["deposit", "default", "housing", "loan"]
    for col in binarias:
        df[col] = df[col].map({"yes": 1, "no": 0})
    categ_nomes = ["job", "marital", "education", "contact", "month", "poutcome"]
    df_encoded = pd.get_dummies(df, columns=categ_nomes, drop_first=True)

    return df_encoded


def padronizar_colunas(X_train, X_test, num_cols):
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, scaler
