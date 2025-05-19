import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.generic import TemplateView
import joblib
import numpy as np

# Charger la liste des communes
data = pd.read_csv('../data/donnees_communes.csv')
communes = sorted(data['Commune'].unique())

class PredictView(APIView):
    def post(self, request):
        try:
            data = request.data
            commune = data.get('commune')
            annee = data.get('annee')
            
            # Validation des entrées
            if not commune or not annee:
                return Response(
                    {"error": "Les champs 'commune' et 'annee' sont requis"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if commune not in communes:
                return Response(
                    {"error": f"Commune '{commune}' non reconnue"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                annee = int(annee)
                if annee < 2024 or annee > 2100:
                    raise ValueError
            except ValueError:
                return Response(
                    {"error": "L'année doit être un entier entre 2024 et 2100"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Charger les modèles et le scaler
            model_recettes = joblib.load(f'../ml_model/models/{commune.lower()}_recettes_model.pkl')
            model_depenses = joblib.load(f'../ml_model/models/{commune.lower()}_depenses_model.pkl')
            scaler = joblib.load(f'../ml_model/scalers/{commune.lower()}_scaler.pkl')
            
            # Préparer les données pour la prédiction
            X = np.array([[annee]])
            X_scaled = scaler.transform(X)
            
            # Faire les prédictions
            recettes = model_recettes.predict(X_scaled)[0]
            depenses = model_depenses.predict(X_scaled)[0]
            
            return Response(
                {
                    "commune": commune,
                    "annee": annee,
                    "recettes": round(recettes, 2),
                    "depenses": round(depenses, 2)
                },
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            return Response(
                {"error": f"Erreur serveur: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PredictFormView(TemplateView):
    template_name = 'prediction/predict.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['communes'] = communes
        return context