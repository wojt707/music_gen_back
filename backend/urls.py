from django.urls import include, path

urlpatterns = [
    path("api/", include("midi_generator.urls")),
]
