from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os

midi_path = r"C:\projects\studia\POLSLrepo_sem7\music_generator\data\ready_midi\Rock\Soft Rock\Rick Astley\Never Gonna Give You Up\TRAXLZU12903D05F94.mid"


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
                {"code": f.name.lower(), "name": f.name}
                for f in os.scandir(
                    r"C:\projects\studia\POLSLrepo_sem7\music_generator\data\ready_midi"
                )
                if f.is_dir()
            ]
            return JsonResponse(genres, safe=False)
        else:
            return JsonResponse({"error": "Wrong method."}, status=405)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
