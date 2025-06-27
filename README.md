# Desafio Técnico – DATA ENGINEER JR (Backend Python)

Este projeto tem como objetivo analisar dados de marketing bancário e construir modelos preditivos para identificar potenciais clientes que possuam maior probabilidade de adesão a uma campanha de marketing.

---

## Como rodar o projeto

### Instalar as dependências

```
pip install -r requirements.txt
```

### (Opcional) Criar ambiente virtual

```
python -m venv venv  # Windows
venv\Scripts\activate
```

### Executar análises exploratórias no terminal

```
python src/eda.py
```

### Treinar o modelo

```
python src/train_model.py
```

### (Opcional) Rodar aplicação Streamlit

```
streamlit run app/app.py

```

## Rodar com Docker (alternativa mais simples)

```
docker build -t bank-marketing-app .
docker run -p 8501:8501 bank-marketing-app

```
