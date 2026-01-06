# main.py
"""
Pipeline KEEL : Prétraitement, nettoyage, conversion CSV↔DAT,
création de folds et discrétisation.

Étapes :
- Chargement CSV / DAT
- Nettoyage numérique et normalisation <LD / NaN
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
INPUT_FILE = "data/iris0.dat"  # Fichier d'entrée (CSV ou DAT)
CLEANED_FILE = "data/data_clean_final.csv"
CLEANED_INTER_FILE = "data/data_clean_check.csv"

FOLD_METHOD = "bloc"      # 'random' ou 'bloc'
NB_EXECUTION = 2
NB_BINS = 10

NON_NUMERIC_COLS = ["ID", "#2", "#3", "#4", "#5"]
TARGET_COL = "Class"

# =====================================================================
# Programme principal
# =====================================================================
def main():
    print(f"[INFO] Début du pipeline. Fichier d'entrée : {INPUT_FILE}")

    # ==========================================================
    # Étape 1 : DAT → CSV si nécessaire
    # ==========================================================
    input_file = INPUT_FILE
    if input_file.lower().endswith(".dat"):
        csv_file = input_file.replace(".dat", ".csv")
        dat_to_csv(input_file, csv_file)
        input_file = csv_file
        print(f"[INFO] Conversion DAT → CSV effectuée : {csv_file}")

    # ==========================================================
    # Étape 2 : Chargement et nettoyage
    # ==========================================================
    print("[INFO] Chargement du fichier CSV")
    df = pd.read_csv(input_file)

    print("[INFO] Détermination des colonnes numériques")
    feature_cols = [c for c in df.columns if c not in NON_NUMERIC_COLS + [TARGET_COL]]

    print("[INFO] Normalisation des valeurs <LD / NaN par colonne numérique")
    df[feature_cols] = normalize_ld_by_column(df[feature_cols], target_col=TARGET_COL)

    print("[INFO] Conversion explicite en float des colonnes numériques")
    df[feature_cols] = df[feature_cols].astype(float)

    # ==========================================================
    # Étape 3 : Suppression des colonnes non numériques
    # ==========================================================
    print("[INFO] Suppression des colonnes non numériques")
    df = df[feature_cols + [TARGET_COL]]  # garder seulement features + target
    print(df.describe())

    # ==========================================================
    # Étape 4 : CSV → DAT
    # ==========================================================
    print("[INFO] Conversion CSV → DAT")
    fichier_dat = csv_to_dat(CLEANED_FILE)
    print(f"[INFO] Fichier DAT généré : {fichier_dat}")

    # ==========================================================
    # Étape 5 : Création des folds
    # ==========================================================
    print(f"[INFO] Création des folds ({FOLD_METHOD})")
    if FOLD_METHOD.lower() == "random":
        header, folds = create_folds_random(fichier_dat, NB_EXECUTION)
    else:
        header, folds = create_folds_bloc(fichier_dat, NB_EXECUTION)
    print(f"[INFO] Nombre de folds générés : {len(folds)}")

    numeric_header = header[:-1]
    class_name = header[-1]

    # ==========================================================
    # Étape 6 : Discrétisation et fichiers KEEL
    # ==========================================================
    for fold_idx, (train_rows, test_rows) in enumerate(folds, start=1):
        print(f"[INFO] Traitement du fold {fold_idx}/{len(folds)}")

        fold_dir = f"data/fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)
        cutpoints_file = f"{fold_dir}/cutpoints_{fold_idx}.txt"

        intervals, train_discretized = discretisation(
            train_rows, numeric_header, NB_BINS, cutpoints_file
        )

        # Discrétisation test
        test_discretized = []
        for row in test_rows:
            values = []
            for val in row[:-1]:
                try:
                    values.append(float(val))
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
                    new_row.append("0")  # fallback si aucune bin match
            test_discretized.append(new_row + [label])

        # Écriture fichiers KEEL
        train_file = f"{fold_dir}/fold_{fold_idx}.training"
        test_file = f"{fold_dir}/fold_{fold_idx}.test"
        desc_file = f"{fold_dir}/fold_{fold_idx}.desc"

        creation_train_test(train_discretized, train_file, intervals)
        creation_train_test(test_discretized, test_file, intervals)
        creation_desc(intervals, numeric_header, class_name, desc_file)

        print(f"[INFO] Fold {fold_idx} terminé")

    print("[INFO] Pipeline terminé avec succès.")

# =====================================================================
# Entrée du script
# =====================================================================
if __name__ == "__main__":
    main()
