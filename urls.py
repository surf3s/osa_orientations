from django.urls import path
from orientations import views as orientations_views

urlpatterns = [
    path('', orientations_views.orientations, name='orientations'),
    path('orientations_r', orientations_views.orientations_r, name='orientations_r'),
    path('callback/<str:endpoint>/', orientations_views.callback, name='callback'),
]
