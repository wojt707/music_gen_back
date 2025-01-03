from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .generator import generate_midi_file


@csrf_exempt
def generate_midi(request):
    """
    Handles MIDI file generation requests.
    """
    # TODO handle errors and bad requests
    try:
        if request.method != "POST":
            return JsonResponse({"message": "Wrong method."}, status=405)

        data = json.loads(request.body)
        # TODO here we cannot set default - it has to be from frontend or error otherwise
        genre = str(data.get("genre", ""))
        bpm = int(data.get("bpm", 120))
        length = int(data.get("length", 100))
        randomness = float(data.get("randomness", 0.0))

        print(f"Requested genre: {genre}")
        midi_path = generate_midi_file(genre, bpm, length, randomness)
        print(midi_path)

        return FileResponse(
            open(midi_path, "rb"),
            as_attachment=True,
            filename=f"{genre}.mid",
        )
    except Exception as e:
        print(e)
        return JsonResponse({"message": str(e)}, status=500)


def get_genres(request):
    """
    Returns the list of available genres.
    """
    try:
        if request.method != "GET":
            return JsonResponse({"message": "Wrong method."}, status=405)
        genres = [
            {"code": "ambient", "name": "Ambient"},
            {"code": "blues", "name": "Blues"},
            {"code": "classical", "name": "Classical"},
            {"code": "country", "name": "Country"},
            {"code": "electronic", "name": "Electronic"},
            {"code": "folk", "name": "Folk"},
            {"code": "jazz", "name": "Jazz"},
            {"code": "latin", "name": "Latin"},
            {"code": "pop", "name": "Pop"},
            {"code": "rap", "name": "Rap"},
            {"code": "rock", "name": "Rock"},
            {"code": "soul", "name": "Soul"},
            {"code": "soundtracks", "name": "Soundtracks"},
            {"code": "world", "name": "World"},
        ]
        return JsonResponse(genres, safe=False)

    except Exception as e:
        print(e)
        return JsonResponse({"message": str(e)}, status=500)
