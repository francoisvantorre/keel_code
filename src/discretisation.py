
from sklearn.model_selection import train_test_split
import numpy as np
import os


def clean_features(features):
    """
    Nettoie les lignes de données numériques en conservant uniquement celles
    ayant le nombre maximal de colonnes.  
    Les lignes incorrectes sont ignorées.

    Paramètres
    ----------
    features : list[list[float]]
        Lignes de données numériques.

    Retour
    ------
    list[list[float]]
        Lignes de données nettoyées.
    """
    max_cols = max(len(row) for row in features)
    min_cols = min(len(row) for row in features)

    print(f"Colonnes minimales détectées : {min_cols}, colonnes maximales : {max_cols}")

    if min_cols != max_cols:
        print("Certaines lignes ont un nombre de colonnes incohérent et seront ignorées.")

    clean = []
    for i, row in enumerate(features):
        if len(row) != max_cols:
            print(f"Ligne ignorée (index={i}, colonnes={len(row)}).")
            continue
        clean.append(row)

    return clean

def compute_intervals_supervised(
    features,
    labels,
    header,
    logfile_path,
    min_samples_per_interval=5,
    min_cut_distance=1e-6,
    max_cuts_per_column=None
):
    """
    Calcule des intervalles de discrétisation supervisée, colonne par colonne.
    Les cutpoints sont créés chaque fois que la variable cible change
    dans les données triées pour chaque colonne.

    Les cutpoints sont limités par min_samples_per_interval, min_cut_distance
    et éventuellement max_cuts_per_column.

    Chaque cutpoint écrit dans le log contient :
        - l’indice dans les données triées
        - la valeur du cutpoint
        - le nom de la colonne
    """

    n_samples = len(features)
    n_features = len(features[0])
    intervals = []

    # UTF-8 pour garder les accents
    with open(logfile_path, "w", encoding="utf-8") as log:
        log.write("DISCRÉTISATION SUPERVISÉE — POINTS DE COUPURE\n")
        log.write("=" * 50 + "\n\n")

        for col_idx in range(n_features):
            feature_name = header[col_idx]

            # Coupler (valeur, label) pour le tri
            col_data = [(features[i][col_idx], labels[i]) for i in range(n_samples)]
            col_data.sort(key=lambda x: x[0])

            col_intervals = []
            cutpoints = []

            start = col_data[0][0]
            prev_val, prev_label = col_data[0]
            samples_since_last_cut = 1

            for sorted_idx in range(1, n_samples):
                val, label = col_data[sorted_idx]
                samples_since_last_cut += 1

                # Changement de classe → candidate pour cut
                if label != prev_label:
                    if samples_since_last_cut >= min_samples_per_interval:
                        cut = (prev_val + val) / 2.0
                        if not cutpoints or abs(cut - cutpoints[-1][1]) >= min_cut_distance:
                            col_intervals.append((start, cut))
                            # stocke l'indice dans les données triées
                            cutpoints.append((sorted_idx, cut))
                            start = cut
                            samples_since_last_cut = 0

                            if max_cuts_per_column is not None and len(cutpoints) >= max_cuts_per_column:
                                break

                prev_val, prev_label = val, label

            col_intervals.append((start, col_data[-1][0]))
            intervals.append(col_intervals)

            # Log : chaque cutpoint sur une ligne
            log.write(f"Variable {col_idx + 1} : {feature_name}\n")
            if cutpoints:
                for idx, val in cutpoints:
                    log.write(f"  coupure à l'indice trié {idx} (valeur = {val:.6f})\n")
            else:
                log.write("  aucun point de coupure\n")
            log.write("\n")

    return intervals
