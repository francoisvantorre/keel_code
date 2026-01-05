import pandas as pd
import numpy as np
import random as rd


# =====================================================================
# Gestion des valeurs <LD
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


# =====================================================================
# Chargement et préparation des données
# =====================================================================



# Chargement des données brutes
data_init = pd.read_csv('data/data_ano.csv')

# Suppression des colonnes non pertinentes
data_num = data_init.drop(["ID", "#2", "#3", "#4", "#5"], axis=1)


# Extraction de la variable cible
data_type = data_num["Type"]

# Suppression des colonnes inutiles pour l'entraînement
data_prelev_1 = data_num[data_num["Prelevement"] == 1]
data_prelev_2 = data_num[data_num["Prelevement"] == 2]

# Remplacement robuste des valeurs <LD
data_num = data_num.applymap(normalize_ld)

# Conversion en numérique
data_num = data_num.apply(pd.to_numeric, errors='coerce')


# Conversion en numérique
data_num = data_num.apply(pd.to_numeric, errors='coerce')


# =====================================================================
# Fonctions de brouillage
# =====================================================================

def choix_numero_brouiller_prob(data, p):
    """
    Sélectionne aléatoirement un nombre de colonnes à brouiller en fonction
    d'une probabilité p.

    Paramètres
    ----------
    data : pandas.DataFrame
        Jeu de données concerné.
    p : float
        Probabilité (entre 0 et 1) déterminant la proportion de colonnes à brouiller.

    Retour
    ------
    list[int]
        Indices des colonnes sélectionnées.
    """
    _, n_colonnes = data.shape
    num = int(n_colonnes * p)
    indices = rd.sample(range(n_colonnes), num)
    return indices


def choix_numero_brouiller_num(data, num):
    """
    Sélectionne aléatoirement un nombre fixe de colonnes à brouiller.

    Paramètres
    ----------
    data : pandas.DataFrame
        Jeu de données concerné.
    num : int
        Nombre de colonnes à brouiller.

    Retour
    ------
    list[int]
        Liste des indices des colonnes sélectionnées.
    """
    _, n_colonnes = data.shape
    indices = rd.sample(range(n_colonnes), num)
    return indices

def normaliser(data, liste, sigma=0.1):
    """
    Ajoute un bruit gaussien proportionnel à la valeur :
        bruit = valeur * N(0, sigma)
    Ainsi, les petites valeurs reçoivent un petit bruit,
    et les grandes valeurs reçoivent un plus grand bruit.
    """
    data_copy = data.copy()

    for col_idx in liste:
        col = data_copy.columns[col_idx]

        # Conversion propre
        x = pd.to_numeric(data_copy[col], errors='coerce')

        # Bruit proportionnel
        bruit = x * np.random.normal(0, sigma, size=len(x))

        # Pour éviter un bruit nul quand x = 0 → bruit = valeur_min * ...
        # (évite que les 0 soient complètement figés)
        min_nonzero = x[x > 0].min() if (x > 0).any() else 0
        bruit[x == 0] = min_nonzero * np.random.normal(0, sigma)

        data_copy[col] = x + bruit

    return data_copy


def brouillage(nom_fichier, data, prob):
    """
    Réalise le brouillage complet d'un jeu de données en sélectionnant aléatoirement
    un nombre donné de colonnes à perturber, puis applique le bruit gaussien.

    Paramètres
    ----------
    nom_fichier : str
        Préfixe du nom du fichier de sortie (sans extension).
    data : pandas.DataFrame
        Données à brouiller.
    num : int
        Nombre de colonnes à brouiller.

    Retour
    ------
    pandas.DataFrame
        Jeu de données final prêt à être exporté.
    """
    # Sélection des colonnes à brouiller
    liste_indices = choix_numero_brouiller_prob(data, prob)

    # Brouillage des colonnes sélectionnées
    data_temp = normaliser(data, liste_indices)

    # Ajout de la colonne cible
    data_fin = pd.concat([data_temp, data_type], axis=1)

    # Export CSV
    data_fin.to_csv(f"data/{nom_fichier}.csv", index=False)

    return data_fin


# =====================================================================
# Exécution
# =====================================================================

# Brouillage des données du prélèvement 1
data_fin = brouillage("data_fin", data_num, 1)
print("Brouillage du prélèvement 1 terminé et exporté sous 'data_fin.csv'")


