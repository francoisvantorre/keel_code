import numpy as np
import pandas as pd

def normalize_ld(x):
    """
    Normalise les valeurs numériques :
    - "<LD" → 0.0
    - NaN / nan / None / vide → 0.0
    - numérique → float
    - autre texte → NaN
    """
    # NaN pandas / None
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0

    # Chaîne de caractères
    if isinstance(x, str):
        s = x.strip()

        if s == "" or s.lower() == "nan" or s == "<LD":
            return 0.0

        try:
            return float(s)
        except ValueError:
            return np.nan

    # Numérique
    if isinstance(x, (int, float)):
        return float(x)

    return np.nan

def normalize_ld_by_column(df, target_col):
    """
    Applique la normalisation des valeurs <LD et NaN sur toutes les colonnes
    sauf la colonne cible.
    """
    for col in df.columns:
        if col != target_col:
            df[col] = df[col].apply(normalize_ld)
    return df
