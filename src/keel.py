from sklearn.model_selection import KFold
import pandas as pd
import os
from discretisation import clean_features, compute_intervals


def colonne_sparse(csv_path, output_path=None, threshold=0.3):
    """
    Supprime les colonnes ayant un pourcentage de valeurs manquantes supérieur
    au seuil spécifié.

    Paramètres
    ----------
    csv_path : str
        Chemin du fichier CSV d'entrée.
    output_path : str, optionnel
        Chemin du fichier CSV nettoyé. Si None, remplace l'original.
    threshold : float
        Seuil maximal autorisé de valeurs manquantes (0,30 = 30%).

    Retour
    ------
    str
        Chemin du fichier CSV nettoyé.
    """
    df = pd.read_csv(csv_path)

    missing_ratio = df.isna().mean()
    cols_to_keep = missing_ratio[missing_ratio <= threshold].index.tolist()
    df_clean = df[cols_to_keep]

    if output_path is None:
        output_path = csv_path

    df_clean.to_csv(output_path, index=False)

    print("Colonnes supprimées :")
    print(list(set(df.columns) - set(cols_to_keep)))
    print(f"Fichier enregistré à : {output_path}")

    return output_path


def csv_to_dat(fichier_csv):
    """
    Convertit un fichier CSV en fichier DAT avec espaces comme séparateurs.

    Paramètres
    ----------
    fichier_csv : str

    Retour
    ------
    str
        Chemin du fichier DAT généré.
    """
    fichier_dat = fichier_csv.replace('.csv', '.dat')

    with open(fichier_csv, 'r') as f_csv, open(fichier_dat, 'w') as f_dat:
        for ligne in f_csv:
            ligne = ligne.strip()
            if not ligne:
                continue
            elements = ligne.split(',')
            f_dat.write(' '.join(elements) + '\n')

    return fichier_dat


def create_folds_random(fichier_dat, k):
    """
    Crée k folds mélangés (KFold avec shuffle).

    Paramètres
    ----------
    fichier_dat : str
        Chemin du fichier DAT.
    k : int
        Nombre de folds.

    Retour
    ------
    tuple
        (header, liste de couples (train_rows, test_rows)).
    """
    with open(fichier_dat, 'r') as f:
        lignes = f.readlines()

    header = lignes[0].strip().split()
    data = [l.strip().split() for l in lignes[1:]]

    folds = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(data):
        train = [data[i] for i in train_idx]
        test = [data[i] for i in test_idx]
        folds.append((train, test))

    return header, folds

    import random
import random

def create_folds_bloc(fichier_dat, n=1):
    """
    Mélange les données et crée 5 folds (block cross-validation).
    Si n > 1, répète la procédure n fois, en refaisant un mélange à chaque tour.
    
    Paramètres :
        fichier_dat : str
            Chemin du fichier .dat
        n : int
            Nombre de répétitions de la création des 5 folds

    Retour :
        header : liste des attributs
        folds  : liste de tuples (train, test)
                 Longueur = 5 * n
    """

    with open(fichier_dat, 'r') as f:
        lignes = f.readlines()

    header = lignes[0].strip().split()
    data_original = [l.strip().split() for l in lignes[1:]]

    all_folds = []

    for iteration in range(n):

        data = data_original.copy()

        random.seed(123 + iteration)
        random.shuffle(data)

        # Taille des blocs
        n_data = len(data)
        block_size = n_data // 5

        # Découpage en 5 blocs
        blocs = []
        for i in range(5):
            start = i * block_size
            end = n_data if i == 4 else (i + 1) * block_size
            blocs.append(data[start:end])

        # Création des 5 folds
        for i in range(5):
            test = blocs[i]
            train = []

            for j in range(5):
                if j != i:
                    train.extend(blocs[j])

            all_folds.append((train, test))

    return header, all_folds




def discretisation(rows, header, nb_bins, logfile_path):
    """
    Discrétise les données d'entraînement à l'aide des intervalles calculés.
    Les lignes contenant des valeurs non numériques sont forcées à 0.0.
    """

    # Séparation features / labels
    raw_features = [r[:-1] for r in rows]
    raw_labels = [r[-1] for r in rows]

    features = []
    labels = []

    # Conversion sécurisée en float (valeurs impossibles → 0.0)
    for i, row in enumerate(raw_features):
        row_float = []
        for val in row:
            try:
                row_float.append(float(val))
            except ValueError:
                row_float.append(0.0)
        features.append(row_float)
        labels.append(raw_labels[i])

    # Nettoyage des lignes incohérentes (colonnes incomplètes)
    features = clean_features(features)

    # Si clean_features supprime tout, on garde quand même au moins une ligne de 0
    if len(features) == 0 and len(rows) > 0:
        n_cols = len(rows[0]) - 1
        features = [[0.0]*n_cols]
        labels = [rows[0][-1]]

    # Calcul des intervalles
    intervals = compute_intervals(features, nb_bins, header, logfile_path)

    # Discrétisation
    rows_discretized = []
    for i, row in enumerate(features):
        new_row = []
        for col_idx, val in enumerate(row):
            for bin_idx, (a, b) in enumerate(intervals[col_idx]):
                if a <= val <= b:
                    new_row.append(str(bin_idx))
                    break
            else:
                new_row.append("0")  # fallback si aucune bin match
        rows_discretized.append(new_row + [labels[i]])

    return intervals, rows_discretized

def creation_desc(intervals, header, class_name, desc_path):
    """
    Crée un fichier .desc compatible KEEL.

    Paramètres
    ----------
    intervals : list
        Intervalles de discrétisation.
    header : list[str]
        Noms des attributs.
    class_name : str
        Nom de la classe.
    desc_path : str
        Chemin du fichier .desc.

    Retour
    ------
    str
        Chemin du fichier généré.
    """
    os.makedirs(os.path.dirname(desc_path), exist_ok=True)

    with open(desc_path, "w") as f:
        for col_idx, col_intervals in enumerate(intervals):
            col_name = header[col_idx]

            interval_strings = []
            for a, b in col_intervals:
                a_str = "-inf" if a is None else round(a, 1)
                b_str = "inf" if b is None else round(b, 1)
                interval_strings.append(f"'({a_str}-{b_str}]'")

            f.write(f"@attribute {col_name} {{{','.join(interval_strings)}}}\n")

        f.write(f"@attribute {class_name} {{positive,negative}}\n")
        f.write(f"@prediction {class_name} = positive\n")

    return desc_path


def creation_train_test(rows_discretized, fichier_out, intervals):
    """
    Génère un fichier .training ou .test compatible KEEL.

    Paramètres
    ----------
    rows_discretized : list[list]
        Lignes discrétisées.
    fichier_out : str
        Chemin du fichier de sortie.
    intervals : list
        Intervalles utilisés.

    Retour
    ------
    str
        Chemin du fichier généré.
    """
    nb_individuals = len(rows_discretized)
    expected_cols = len(intervals) + 1

    class_map = {"malade": "positive", "sain": "negative"}

    with open(fichier_out, 'w') as f:
        f.write(f"Nb Individuals: {nb_individuals}\n")

        for row in rows_discretized:

            if len(row) < expected_cols:
                missing = expected_cols - len(row)
                row = row[:-1] + ["0"] * missing + [row[-1]]

            elif len(row) > expected_cols:
                row = row[:expected_cols]

            line_items = []
            for col_idx, val in enumerate(row[:-1]):
                val_int = int(val)

                if val_int >= len(intervals[col_idx]):
                    val_int = len(intervals[col_idx]) - 1

                low, high = intervals[col_idx][val_int]
                low_str = "-inf" if low is None else str(low)
                high_str = "inf" if high is None else str(high)

                line_items.append(f"{col_idx} '({low_str}-{high_str}]'")

            original_class = row[-1]
            mapped_class = class_map.get(original_class, original_class)
            line_items.append(f"{len(row)-1} {mapped_class}")

            f.write("{" + ",".join(line_items) + "}\n")

    return fichier_out
