from django.urls import path
from . import views

urlpatterns = [
    path('sentiment/', views.InputView.as_view(), name='sentiment'),
]