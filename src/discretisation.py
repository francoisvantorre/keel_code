
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


def compute_intervals(features, nb_bins, header, logfile_path):
    """
    Calcule des intervalles de discrétisation en fréquences égales et écrit
    les points de coupure dans un fichier texte.

    Paramètres
    ----------
    features : list[list[float]]
        Données numériques nettoyées.
    nb_bins : int
        Nombre de classes de discrétisation.
    header : list[str]
        Noms des attributs numériques.
    logfile_path : str
        Chemin du fichier où seront enregistrés les cutpoints.

    Retour
    ------
    list[list[tuple(float, float)]]
        Liste des intervalles pour chaque attribut.
    """

    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

    intervals = []
    nb_cols = len(features[0])

    with open(logfile_path, "w") as logfile:

        for col in range(nb_cols):
            col_name = header[col]
            values = [row[col] for row in features]

            sorted_vals = sorted(values)
            n = len(sorted_vals)
            step = n // nb_bins if nb_bins > 0 else 1

            frontiere = []
            for b in range(nb_bins + 1):
                idx = min(b * step, n - 1)
                val = sorted_vals[idx]
                frontiere.append(val)

                line = f"Colonne : {col_name}, bin {b}, index {idx}, valeur {val}\n"
                print(line.strip())
                logfile.write(line)

            logfile.write("\n")

            col_intervals = []
            for i in range(nb_bins):
                a = round(frontiere[i], 1)
                b = round(frontiere[i + 1], 1)
                col_intervals.append((a, b))

            intervals.append(col_intervals)

    return intervals
