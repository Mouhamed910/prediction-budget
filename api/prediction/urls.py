from django.urls import path
from .views import PredictView, PredictFormView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('', PredictFormView.as_view(), name='predict_form'),
]