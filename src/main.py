# main.py
"""
Pipeline KEEL :
- Chargement CSV / DAT
- Nettoyage numérique (<LD, NaN → 0.0)
- Suppression des colonnes non numériques
- Conversion CSV → DAT
- Création des folds
- Discrétisation et génération des fichiers KEEL
"""

import os
import pandas as pd

from nettoyage import normalize_ld_by_column
from dat_to_csv import dat_to_csv
from keel import (
    csv_to_dat,
    create_folds_random,
    create_folds_bloc,
    discretisation,
    creation_desc,
    creation_train_test
)

# =====================================================================
# Configuration
# =====================================================================
# INPUT_FILE = "data/iris0.dat"     # CSV ou DAT
INPUT_FILE = "data/data_ano.csv"
CLEANED_FILE = "data/data_clean_final.csv"

FOLD_METHOD = "random"              # 'random' ou 'bloc'
NB_EXECUTION = 2                  # LE nb de folds sera NB_EXECUTION * 5
NB_BINS = 10

NON_NUMERIC_COLS = ["ID", "#2", "#3", "#4", "#5","Prelevement"]
TARGET_COL = "Type"

# =====================================================================
# Programme principal
# =====================================================================
def main():

    print(f"[INFO] Début du pipeline : {INPUT_FILE}")

    # ==========================================================
    # 1. DAT → CSV si nécessaire
    # ==========================================================
    input_file = INPUT_FILE
    if input_file.lower().endswith(".dat"):
        csv_file = input_file.replace(".dat", ".csv")
        dat_to_csv(input_file, csv_file)
        input_file = csv_file
        print(f"[INFO] Conversion DAT → CSV : {csv_file}")

    # ==========================================================
    # 2. Chargement CSV
    # ==========================================================
    print("[INFO] Chargement du CSV")
    df = pd.read_csv(input_file)

    # ==========================================================
    # 3. Détection des colonnes numériques
    # ==========================================================
    feature_cols = [
        c for c in df.columns
        if c not in NON_NUMERIC_COLS + [TARGET_COL]
    ]

    print(f"[INFO] Colonnes numériques détectées : {len(feature_cols)}")

    # ==========================================================
    # 4. Nettoyage <LD / NaN → 0.0
    # ==========================================================
    print("[INFO] Nettoyage <LD / NaN")
    df[feature_cols] = normalize_ld_by_column(df[feature_cols])

    # Conversion explicite en float
    df[feature_cols] = df[feature_cols].astype(float)

    # ==========================================================
    # 5. Suppression des colonnes non numériques
    # ==========================================================
    print("[INFO] Suppression des colonnes non numériques")
    df = df[feature_cols + [TARGET_COL]]

    # ==========================================================
    # 6. Sauvegarde CSV nettoyé (DEBUG / CHECK)
    # ==========================================================
    print("[INFO] Sauvegarde du CSV nettoyé")
    df.to_csv(CLEANED_FILE, index=False)

    print("[INFO] Aperçu statistiques :")
    print(df[feature_cols].describe())

    # ==========================================================
    # 7. CSV → DAT
    # ==========================================================
    print("[INFO] Conversion CSV → DAT")
    fichier_dat = csv_to_dat(CLEANED_FILE)
    print(f"[INFO] Fichier DAT généré : {fichier_dat}")

    # ==========================================================
    # 8. Création des folds
    # ==========================================================
    print(f"[INFO] Création des folds ({FOLD_METHOD})")
    if FOLD_METHOD == "random":
        header, folds = create_folds_random(fichier_dat, NB_EXECUTION)
    else:
        header, folds = create_folds_bloc(fichier_dat, NB_EXECUTION)

    print(f"[INFO] Nombre de folds : {len(folds)}")

    numeric_header = header[:-1]
    class_name = header[-1]

    # ==========================================================
    # 9. Discrétisation + fichiers KEEL
    # ==========================================================
    for fold_idx, (train_rows, test_rows) in enumerate(folds, start=1):

        print(f"[INFO] Fold {fold_idx}/{len(folds)}")

        fold_dir = f"data/fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)

        cutpoints_file = f"{fold_dir}/cutpoints_{fold_idx}.txt"

        intervals, train_discretized = discretisation(
            train_rows,
            numeric_header,
            NB_BINS,
            cutpoints_file
        )

        # --- Discrétisation test ---
        test_discretized = []
        for row in test_rows:
            values = []
            for v in row[:-1]:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(0.0)

            label = row[-1]
            new_row = []

            for col_idx, val in enumerate(values):
                for i, (a, b) in enumerate(intervals[col_idx]):
                    if a <= val <= b:
                        new_row.append(str(i))
                        break
                else:
                    new_row.append("0")

            test_discretized.append(new_row + [label])

        # --- Écriture fichiers KEEL ---
        train_file = f"{fold_dir}/fold_{fold_idx}.training"
        test_file = f"{fold_dir}/fold_{fold_idx}.test"
        desc_file = f"{fold_dir}/fold_{fold_idx}.desc"

        creation_train_test(train_discretized, train_file, intervals)
        creation_train_test(test_discretized, test_file, intervals)
        creation_desc(intervals, numeric_header, class_name, desc_file)

        print(f"[INFO] Fold {fold_idx} terminé")

    print("[INFO] Pipeline terminé avec succès")

# =====================================================================
if __name__ == "__main__":
    main()
