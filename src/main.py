# main.py

import os
from keel import colonne_sparse, csv_to_dat, create_folds_random,create_folds_bloc
from keel import discretisation, creation_desc, creation_train_test

# =====================================================================
# Initalisation des variables
fold = 'bloc'  # 'random' ou 'bloc' selon la méthode de création des folds    
nb_execution = 2  # Nombre d'execution dans les focntions pour créer les folds
#/!\ pour la méthode 'random', le nombre d'execution donne le nombre de folds 
# /!\ pour la méthode 'bloc', le nombre d'execution donne un nombre de folds = 5 * nb_execution
# =====================================================================


def main():
    """
    Programme principal :
    - Nettoyage du CSV
    - Conversion en DAT
    - Création des folds
    - Discrétisation train/test
    - Création des fichiers KEEL (.training / .test / .desc)
    """

    fichier_input = 'data/data_fin.csv'

    fichier_clean = colonne_sparse(
        fichier_input,
        output_path='data/data_fin_clean.csv',
        threshold=0.30
    )

    if fichier_clean.lower().endswith(".csv"):
        fichier_dat = csv_to_dat(fichier_clean)
        print(f"Conversion en DAT effectuée : {fichier_dat}")
    else:
        fichier_dat = fichier_clean

    if fold == 'random':
        header, folds = create_folds_random(fichier_dat, nb_execution)
        print(f"{nb_execution} folds ont été générés.")
    else:
        header, folds = create_folds_bloc(fichier_dat, nb_execution)
        print(f"{nb_execution} folds ont été générés.")

    numeric_header = header[:-1]
    class_name = header[-1]
    nb_bins = 10

    for fold_idx, (train_rows, test_rows) in enumerate(folds, start=1):
        print(f"Traitement du fold {fold_idx}/{nb_execution}")

        fold_dir = f"data/fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)

        cutpoints_file = f"{fold_dir}/cutpoints_{fold_idx}.txt"

        intervals, train_discretized = discretisation(
            train_rows, numeric_header, nb_bins, cutpoints_file
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

        train_file = f"{fold_dir}/fold_{fold_idx}.training"
        test_file = f"{fold_dir}/fold_{fold_idx}.test"

        creation_train_test(train_discretized, train_file, intervals)
        creation_train_test(test_discretized, test_file, intervals)

        desc_file = f"{fold_dir}/fold_{fold_idx}.desc"
        creation_desc(intervals, numeric_header, class_name, desc_file)

        print(f"Fichiers générés pour le fold {fold_idx}.")


if __name__ == "__main__":
    main()
