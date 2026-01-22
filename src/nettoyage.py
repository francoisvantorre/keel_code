import numpy as np
import pandas as pd

def normalize_ld_by_column(df):
    """
    Remplace <LD dans chaque colonne par la moitié du minimum non nul de la colonne.
    Convertit aussi les NaN / None / vide en 0.0.
    """
    df_new = df.copy()

    for col in df_new.columns:
        # Convertir en string pour détecter "<LD>", en ignorant espaces et casse
        col_str = df_new[col].astype(str).str.strip().str.upper()

        # Calculer le minimum non nul, en convertissant les valeurs numériques seulement
        numeric_values = pd.to_numeric(df_new[col], errors='coerce')
        min_nonzero = numeric_values[numeric_values > 0].min()
        min_half = min_nonzero / 2 if pd.notna(min_nonzero) else 0.0

        # Remplacement
        def mapper(x):
            # Si <LD → min_half
            if isinstance(x, str) and x.strip().upper() == "<LD":
                return min_half
            # Si None / NaN / vide → 0
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return 0.0
            # Sinon, convertir en float si possible
            try:
                return float(x)
            except ValueError:
                return np.nan  # texte non convertible

        df_new[col] = df_new[col].map(mapper)

    return df_new

