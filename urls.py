from django.urls import path
from orientations import views as orientations_views

urlpatterns = [
    path('', orientations_views.orientations, name='orientations'),
    path('callback/<str:endpoint>/', orientations_views.callback, name='callback'),
]
