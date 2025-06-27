def conferir_missing(df):
    missing_found = False
    for col in df.columns:
        n_null = df[col].isnull().sum()
        n_unknown = 0
        if df[col].dtype == "object":
            n_unknown = (df[col] == "unknown").sum()

        total_missing = n_null + n_unknown

        if total_missing > 0:
            missing_found = True
            perc = round(100 * total_missing / len(df), 2)
            print(f"{col}: {total_missing} missing ({perc}%)")

    if not missing_found:
        print("Não há missing no banco de dados")


def categorize_corr(val):
    if abs(val) >= 0.7:
        return "Alto"
    elif abs(val) >= 0.4:
        return "Moderado"
    elif abs(val) >= 0.1:
        return "Baixo"
    else:
        return "Nenhum"
