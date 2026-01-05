# main.py
"""
Pipeline KEEL : Prétraitement, nettoyage, conversion CSV↔DAT, création de folds et discrétisation.

Fonctionnalités :
- Conversion DAT → CSV si nécessaire
- Nettoyage des colonnes numériques et traitement des valeurs <LD / NA
- Conversion CSV → DAT (format KEEL)
- Création de folds (méthode bloc ou random)
- Discrétisation train/test et création des fichiers KEEL (.training / .test / .desc)
"""

import os
import pandas as pd
from keel import csv_to_dat, create_folds_random, create_folds_bloc
from keel import discretisation, creation_desc, creation_train_test
from dat_to_csv import dat_to_csv

# =====================================================================
# Configuration
# =====================================================================
INPUT_FILE = "data/data_ano.csv"  # fichier source
CLEANED_FILE = "data/data_clean_final.csv"
FOLD_METHOD = "bloc"  # 'random' ou 'bloc'
NB_EXECUTION = 2       # nombre d'exécutions pour création folds
NB_BINS = 10           # nombre de bins pour discrétisation

# Colonnes qui ne font pas partie des données numériques
NON_NUMERIC_COLS = ["ID", "#2", "#3", "#4", "#5"]

# =====================================================================
# Fonctions utilitaires
# =====================================================================

def normalize_ld(x):
    """
    Normalise les valeurs <LD / LD / NA / vides vers 0.0
    Compatible avec le pipeline KEEL.
    """
    if isinstance(x, str):
        x = x.strip().upper()
        if x in ["<LD", "LD", "NA", ""]:
            return 0.0
    return x

def clean_csv_numeric(input_file, output_file, target_col=None):
    """
    Nettoie le CSV en :
    - Remplaçant les valeurs <LD / LD / NA par 0
    - Convertion en float
    - Suppression des colonnes inutiles
    - Conservation du fichier final nettoyé
    """
    print(f"[INFO] Chargement du fichier : {input_file}")
    df = pd.read_csv(input_file)

    # Nettoyage des valeurs <LD
    print("[INFO] Normalisation des valeurs <LD / NA / vides ...")
    df = df.applymap(normalize_ld)

    # Suppression des colonnes non numériques
    cols_to_drop = [c for c in NON_NUMERIC_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Conversion en numérique
    print("[INFO] Conversion des colonnes en numérique ...")
    df = df.apply(pd.to_numeric, errors="coerce")

    # Vérification de la colonne cible
    if target_col is None or target_col not in df.columns:
        target_col = df.columns[-1]  # fallback sur dernière colonne
    print(f"[INFO] Colonne cible utilisée : {target_col}")

    # Vérifier qu'il reste au moins une colonne numérique
    numeric_cols = df.drop(columns=[target_col]).select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique valide après nettoyage.")

    # Sauvegarde du CSV final
    df.to_csv(output_file, index=False)
    print(f"[INFO] Fichier nettoyé sauvegardé à : {output_file}")

    return df, target_col

# =====================================================================
# Programme principal
# =====================================================================

def main():
    """
    Pipeline principal :
    - Conversion DAT → CSV si nécessaire
    - Nettoyage CSV
    - Conversion CSV → DAT
    - Création des folds
    - Discrétisation train/test
    - Création des fichiers KEEL
    """
    print(f"[INFO] Début du pipeline. Fichier d'entrée : {INPUT_FILE}")

    # =============================
    # Conversion DAT → CSV si nécessaire
    # =============================
    input_file = INPUT_FILE
    if input_file.lower().endswith(".dat"):
        csv_file = input_file.replace(".dat", ".csv")
        dat_to_csv(input_file, csv_file)
        input_file = csv_file
        print(f"[INFO] Conversion DAT → CSV effectuée : {csv_file}")

    # =============================
    # Nettoyage du CSV
    # =============================
    df, TARGET_COL = clean_csv_numeric(input_file, CLEANED_FILE)

    # =============================
    # Conversion CSV → DAT
    # =============================
    print("[INFO] Conversion CSV → DAT ...")
    fichier_dat = csv_to_dat(CLEANED_FILE)
    print(f"[INFO] Conversion effectuée : {fichier_dat}")

    # =============================
    # Création des folds
    # =============================
    print(f"[INFO] Création des folds ({FOLD_METHOD}) ...")
    if FOLD_METHOD.lower() == "random":
        header, folds = create_folds_random(fichier_dat, NB_EXECUTION)
    else:
        header, folds = create_folds_bloc(fichier_dat, NB_EXECUTION)
    print(f"[INFO] {len(folds)} folds générés.")

    numeric_header = header[:-1]
    class_name = header[-1]

    # =============================
    # Discrétisation + génération fichiers KEEL
    # =============================
    for fold_idx, (train_rows, test_rows) in enumerate(folds, start=1):
        print(f"[INFO] Traitement du fold {fold_idx} / {len(folds)}")
        fold_dir = f"data/fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)
        cutpoints_file = f"{fold_dir}/cutpoints_{fold_idx}.txt"

        intervals, train_discretized = discretisation(
            train_rows, numeric_header, NB_BINS, cutpoints_file
        )

        test_discretized = []
        for row in test_rows:
            vals = row[:-1]
            label = row[-1]
            new_row = []
            for col_idx, val in enumerate(vals):
                val = float(val)
                for i, (a, b) in enumerate(intervals[col_idx]):
                    if a <= val <= b:
                        new_row.append(str(i))
                        break
            test_discretized.append(new_row + [label])

        # Fichiers KEEL
        train_file = f"{fold_dir}/fold_{fold_idx}.training"
        test_file = f"{fold_dir}/fold_{fold_idx}.test"
        creation_train_test(train_discretized, train_file, intervals)
        creation_train_test(test_discretized, test_file, intervals)

        desc_file = f"{fold_dir}/fold_{fold_idx}.desc"
        creation_desc(intervals, numeric_header, class_name, desc_file)

        print(f"[INFO] Fichiers générés pour le fold {fold_idx}")

    print("[INFO] Pipeline terminé.")

# =====================================================================
# Entrée du script
# =====================================================================
if __name__ == "__main__":
    main()
