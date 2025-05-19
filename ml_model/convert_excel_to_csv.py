import pandas as pd

# Charger le fichier Excel
data = pd.read_excel('../data/donnees_communes.xlsx')

# Sauvegarder en CSV
data.to_csv('../data/donnees_communes.csv', index=False)
print("Fichier CSV généré : data/donnees_communes.csv")