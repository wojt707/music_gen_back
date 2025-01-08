from django.urls import path

from . import views

urlpatterns = [
    path("genres/", views.get_genres, name="get_genres"),
    path("generate/", views.generate_midi, name="generate_midi"),
    path("samples/<str:sample_name>/", views.get_piano_sample, name="samples"),
]
