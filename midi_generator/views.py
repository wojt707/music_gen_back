from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from pathlib import Path

midi_path = Path(__file__).resolve().parent.parent / "models" / "temp.mid"


@csrf_exempt
def generate_midi(request):
    """
    Handles MIDI file generation requests.
    """
    try:
        if request.method == "POST":
            data = json.loads(request.body)
            genre = data.get("genre", "")
            print(f"Requested genre: {genre}")
            return FileResponse(
                open(midi_path, "rb"), as_attachment=True, filename="generated-midi.mid"
            )
        else:
            return JsonResponse({"error": "Wrong method."}, status=405)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def get_genres(request):
    """
    Returns the list of available genres.
    """
    try:
        if request.method == "GET":
            genres = [
                {"code": "ambient", "name": "Ambient"},
                {"code": "blues", "name": "Blues"},
                {"code": "children", "name": "Children"},
                {"code": "classical", "name": "Classical"},
                {"code": "country", "name": "Country"},
                {"code": "electronic", "name": "Electronic"},
                {"code": "folk", "name": "Folk"},
                {"code": "jazz", "name": "Jazz"},
                {"code": "latin", "name": "Latin"},
                {"code": "pop", "name": "Pop"},
                {"code": "rap", "name": "Rap"},
                {"code": "reggae", "name": "Reggae"},
                {"code": "religious", "name": "Religious"},
                {"code": "rock", "name": "Rock"},
                {"code": "soul", "name": "Soul"},
                {"code": "soundtracks", "name": "Soundtracks"},
                {"code": "unknown", "name": "Unknown"},
                {"code": "world", "name": "World"},
            ]
            return JsonResponse(genres, safe=False)
        else:
            return JsonResponse({"error": "Wrong method."}, status=405)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
