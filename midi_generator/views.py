from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from .generator import generate_midi_file


class CustomFileResponse(FileResponse):
    def __init__(self, *args, as_attachment=..., filename=..., midi_path, **kwargs):
        super().__init__(
            *args, as_attachment=as_attachment, filename=filename, **kwargs
        )
        self.midi_path = midi_path

    def close(self):
        super(CustomFileResponse, self).close()

        try:
            os.remove(self.midi_path)
            print(f"Midi file deleted. {self.midi_path}")
        except Exception as e:
            print(f"Error deleting MIDI file: {e}")


@csrf_exempt
def generate_midi(request):
    """
    Handles MIDI file generation requests.
    """
    if request.method != "POST":
        return JsonResponse({"message": "Wrong method."}, status=405)

    try:
        # Parse and validate the request data
        data = json.loads(request.body)
        genre = data.get("genre")
        bpm = data.get("bpm")
        length = data.get("length")
        randomness = data.get("randomness")

        if not genre or not isinstance(genre, str):
            return JsonResponse({"message": "Invalid or missing genre."}, status=400)
        if not isinstance(bpm, int) or bpm <= 0:
            return JsonResponse({"message": "Invalid BPM."}, status=400)
        if not isinstance(length, int) or length <= 0:
            return JsonResponse({"message": "Invalid length."}, status=400)
        if (
            (not isinstance(randomness, float) and not isinstance(randomness, int))
            or randomness < 0
            or randomness > 1
        ):
            return JsonResponse({"message": "Invalid randomness."}, status=400)

        # Generate the MIDI file
        midi_path = generate_midi_file(genre, bpm, length, randomness)
        if not os.path.exists(midi_path):
            return JsonResponse({"message": "MIDI generation failed."}, status=500)

        return CustomFileResponse(
            open(midi_path, "rb"),
            as_attachment=True,
            filename=f"{genre}.mid",
            midi_path=midi_path,
        )

    except json.JSONDecodeError:
        return JsonResponse({"message": "Invalid JSON format."}, status=400)
    except Exception as e:
        print(f"Error generating MIDI: {e}")
        return JsonResponse({"message": str(e)}, status=500)


def get_genres(request):
    """
    Returns the list of available genres.
    """
    if request.method != "GET":
        return JsonResponse({"message": "Wrong method."}, status=405)
    try:
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
