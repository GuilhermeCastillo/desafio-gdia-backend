import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model_bank.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_bank.pkl")
COLS_PATH = os.path.join(BASE_DIR, "models", "cols_model.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
cols_model = joblib.load(COLS_PATH)

num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

base_categories = {
    "job": "admin.",
    "marital": "divorced",
    "education": "primary",
    "contact": "cellular",
    "month": "apr",
    "poutcome": "sem_contato_anterior",
}

st.set_page_config(page_title="Preditor de Campanha Bancária", layout="centered")
st.title("Predição de Resposta a Campanha de Marketing")

st.markdown(
    "Preencha os dados do cliente abaixo para prever se ele aceitaria a oferta da campanha."
)

age = st.number_input("Idade", min_value=18, max_value=100, value=30)
balance = st.number_input("Saldo bancário", value=0.0, step=0.01, format="%.2f")
day = st.number_input("Dia do mês de contato", min_value=1, max_value=31, value=15)
duration = st.number_input("Duração do último contato (em segundos)", value=100)
campaign = st.number_input("Número de contatos nesta campanha", value=1)
pdays = st.number_input("Dias desde o último contato anterior", value=999)
previous = st.number_input("Número de contatos anteriores", value=0)

job = st.selectbox(
    "Profissão",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
    ],
)

marital = st.selectbox("Estado civil", ["divorced", "married", "single"])
education = st.selectbox("Educação", ["primary", "secondary", "tertiary"])
contact = st.selectbox("Tipo de contato", ["cellular", "telephone"])
month = st.selectbox(
    "Mês do contato",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ],
)
poutcome = st.selectbox(
    "Resultado da campanha anterior",
    ["failure", "other", "success", "sem_contato_anterior"],
)

default = st.radio("Possui inadimplência?", ["no", "yes"])
housing = st.radio("Possui financiamento habitacional?", ["no", "yes"])
loan = st.radio("Possui empréstimo pessoal?", ["no", "yes"])

if st.button("Prever resposta à campanha"):
    # Inicializa dicionário com todas as colunas como 0
    input_dict = {col: 0 for col in cols_model}

    # Adiciona variáveis numéricas e binárias
    input_dict.update(
        {
            "age": age,
            "balance": balance,
            "day": day,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "default": 1 if default == "yes" else 0,
            "housing": 1 if housing == "yes" else 0,
            "loan": 1 if loan == "yes" else 0,
        }
    )

    user_cats = {
        "job": job,
        "marital": marital,
        "education": education,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
    }

    for cat, val in user_cats.items():
        if val != base_categories[cat]:
            col_name = f"{cat}_{val}"
            if col_name in input_dict:
                input_dict[col_name] = 1

    input_data = pd.DataFrame([input_dict])
    input_data = input_data[cols_model]

    try:
        input_data[num_cols] = scaler.transform(input_data[num_cols])
    except Exception as e:
        st.error(f"Erro ao aplicar scaler: {e}")
        st.stop()

    prediction = model.predict(input_data)[0]
    resultado = (
        "**SIM** – O cliente aceitaria a campanha."
        if prediction == 1
        else "**NÃO** – O cliente não aceitaria a campanha."
    )
    st.success(resultado)
