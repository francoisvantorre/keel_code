import csv
import re

def dat_to_csv(dat_file, csv_file):
    attributes = []
    data_rows = []
    data_section = False

    with open(dat_file, "r") as f:
        for line in f:
            line = line.strip()

            # ignorer lignes vides ou commentaires
            if not line or line.startswith("%"):
                continue

            # récupérer les noms d'attributs
            if line.lower().startswith("@attribute"):
                # @attribute Nom type
                match = re.match(r"@attribute\s+(['\"]?)(\w+)\1", line, re.IGNORECASE)
                if match:
                    attributes.append(match.group(2))

            # début des données
            elif line.lower() == "@data":
                data_section = True
                continue

            # lignes de données
            elif data_section:
                values = [v.strip() for v in line.split(",")]
                data_rows.append(values)

    if not attributes:
        raise ValueError("Aucun attribut détecté dans le fichier")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(attributes)
        writer.writerows(data_rows)

    print(f"CSV généré avec succès : {csv_file}")
    return csv_file
