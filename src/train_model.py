import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import (
    carregar_dados,
    tratar_valores_nulos,
    encode_variaveis,
    padronizar_colunas,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bank.xls")
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

df = carregar_dados(DATA_PATH)
df = tratar_valores_nulos(df)
df_encoded = encode_variaveis(df)

X = df_encoded.drop("deposit", axis=1)
y = df_encoded["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
X_train, X_test, scaler = padronizar_colunas(X_train, X_test, num_cols)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)

print("Resultados Teste:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_test):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_test):.4f}")

joblib.dump(model, os.path.join(models_dir, "model_bank.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler_bank.pkl"))
joblib.dump(X.columns.to_list(), os.path.join(models_dir, "cols_model.pkl"))
