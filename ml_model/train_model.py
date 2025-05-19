import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Charger les données
data = pd.read_csv('../data/donnees_communes.csv')

# Créer les dossiers pour sauvegarder les modèles et scalers
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

# Liste des communes
communes = data['Commune'].unique()

for commune in communes:
    print(f"Entraînement pour {commune}...")
    
    # Filtrer les données pour la commune
    commune_data = data[data['Commune'] == commune]
    
    # Features (Année) et cibles (Recettes, Dépenses)
    X = commune_data[['Année']].values
    y_recettes = commune_data['Recettes (M€)'].values
    y_depenses = commune_data['Dépenses (M€)'].values
    
    # Standardiser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entraîner un modèle pour les recettes
    model_recettes = RandomForestRegressor(n_estimators=100, random_state=42)
    model_recettes.fit(X_scaled, y_recettes)
    
    # Entraîner un modèle pour les dépenses
    model_depenses = RandomForestRegressor(n_estimators=100, random_state=42)
    model_depenses.fit(X_scaled, y_depenses)
    
    # Sauvegarder les modèles et le scaler
    joblib.dump(model_recettes, f'models/{commune.lower()}_recettes_model.pkl')
    joblib.dump(model_depenses, f'models/{commune.lower()}_depenses_model.pkl')
    joblib.dump(scaler, f'scalers/{commune.lower()}_scaler.pkl')
    
    print(f"Modèles et scaler pour {commune} sauvegardés.")