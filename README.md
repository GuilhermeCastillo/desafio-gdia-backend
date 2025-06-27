# Desafio Técnico – DATA ENGINEER JR (Backend Python)

## SOLUÇÃO

Instale o DOCKER para executar o projeto!

docker build -t bank-marketing-app .
docker run -p 8501:8501 bank-marketing-app

## 🎯 Objetivo

Avaliar a capacidade de manipular e explorar dados utilizando **Python** com **DuckDB**, além de construir uma pipeline simples de Machine Learning baseada em dados públicos.

## 📦 Dataset

Escolha um dataset do [Kaggle](https://www.kaggle.com/datasets) que seja simples e bem documentado.

> **Dica:** Prefira datasets com colunas categóricas, valores faltantes e tarefa de classificação binária.

## 📌 Etapas do Desafio

### 1. Ingestão e Armazenamento

- Baixe o dataset escolhido do Kaggle.
- Carregue os dados em uma instância local de **DuckDB**, via Python.

### 2. Exploração de Dados (EDA)

Utilize **consultas SQL no DuckDB** para responder:

- Quantas linhas e colunas o dataset possui?
- Existem valores nulos? Em quais colunas e quantos?
- Qual a correlação entre variáveis numéricas?

### 3. Pré-processamento

- Trate valores nulos adequadamente.
- Converta variáveis categóricas para numéricas (One-Hot Encoding ou Ordinal).
- Normalize ou padronize colunas numéricas, se necessário.

### 4. Modelagem

- Faça a divisão entre treino e teste com `train_test_split`.
- Avalie utilizando: **accuracy**, **precision**, **recall** e **F1-score**.

### 5. Apresentação

Você precisará nos apresentar o seu projeto em uma videoconferência de até 15 minutos.
Durante essa apresentação, avaliaremos não apenas as decisões técnicas tomadas, mas também sua:

Capacidade de comunicação
Clareza na explicação
Organização do raciocínio
Comportamento colaborativo e alinhamento com trabalho em equipe

Além do domínio técnico, buscamos alguém que saiba se comunicar bem com diferentes áreas e lidar com os desafios naturais de alinhamento e compartilhamento de informações em ambientes multidisciplinares.

### ## ⏰ Prazo de Entrega

\*\*3 dias corridos\*\* após o envio deste desafio. Caso termina antes, ficamos disponíveis para falar ok?

Boa sorte! 🍀

docker build -t bank-marketing-app .

docker run -p 8501:8501 bank-marketing-app
