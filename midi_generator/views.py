from pathlib import Path
from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from .generator import generate_midi_file


class CustomFileResponse(FileResponse):
    """
    A custom FileResponse class to handle MIDI file responses.
    Automatically deletes the MIDI file after the response is closed.
    """

    def __init__(
        self,
        *args,
        as_attachment: bool = ...,
        filename: str = ...,
        midi_path: str,
        **kwargs,
    ):
        """
        Initialize the custom file response.

        Parameters:
        - as_attachment (bool): Whether to treat the file as an attachment.
        - filename (str): The name of the file to be downloaded.
        - midi_path (str): The path to the MIDI file.
        """
        super().__init__(
            *args, as_attachment=as_attachment, filename=filename, **kwargs
        )
        self.midi_path = midi_path

    def close(self):
        """
        Closes the response and deletes the associated MIDI file.
        """
        super(CustomFileResponse, self).close()
        try:
            os.remove(self.midi_path)
            print(f"Midi file deleted. {self.midi_path}")
        except Exception as e:
            print(f"Error deleting MIDI file: {e}")


@csrf_exempt
def generate_midi(request: HttpResponse) -> JsonResponse | CustomFileResponse:
    """
    Handles MIDI file generation requests.

    Parameters:
    - request (HttpResponse): The incoming HTTP request.

    Returns:
    - JsonResponse: Error or success messages.
    - CustomFileResponse: MIDI file response upon successful generation.
    """
    if request.method != "POST":
        return JsonResponse({"message": "Wrong method."}, status=405)

    try:
        # Parse and validate the request data
        data: dict = json.loads(request.body)
        genre: str = data.get("genre")
        bpm: int = data.get("bpm")
        length: int = data.get("length")
        randomness: float = data.get("randomness")

        if not genre or not isinstance(genre, str):
            return JsonResponse({"message": "Invalid or missing genre."}, status=400)
        if not isinstance(bpm, int) or bpm <= 0:
            return JsonResponse({"message": f"Invalid BPM. {bpm}"}, status=400)
        if not isinstance(length, int) or length <= 0:
            return JsonResponse({"message": f"Invalid length. {length}"}, status=400)
        if (
            (not isinstance(randomness, (int, float)))
            or randomness < -100
            or randomness > 100
        ):
            return JsonResponse(
                {"message": f"Invalid randomness. {randomness}"}, status=400
            )

        # Generate the MIDI file
        midi_path: str = generate_midi_file(genre, bpm, length, float(randomness))
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


def get_genres(request: HttpResponse) -> JsonResponse:
    """
    Returns the list of available genres.

    Parameters:
    - request (HttpResponse): The incoming HTTP request.

    Returns:
    - JsonResponse: A list of genre dictionaries or error messages.
    """
    if request.method != "GET":
        return JsonResponse({"message": "Wrong method."}, status=405)

    try:
        genres: list[dict] = [
            {"code": "ambient", "name": "Ambient", "bpm": 136},
            {"code": "blues", "name": "Blues", "bpm": 120},
            {"code": "children", "name": "Children", "bpm": 100},
            {"code": "classical", "name": "Classical", "bpm": 60},
            {"code": "country", "name": "Country", "bpm": 60},
            {"code": "electronic", "name": "Electronic", "bpm": 120},
            {"code": "folk", "name": "Folk", "bpm": 164},
            {"code": "jazz", "name": "Jazz", "bpm": 92},
            {"code": "latin", "name": "Latin", "bpm": 56},
            {"code": "pop", "name": "Pop", "bpm": 106},
            {"code": "rap", "name": "Rap", "bpm": 89},
            {"code": "reggae", "name": "Reggae", "bpm": 129},
            {"code": "religious", "name": "Religious", "bpm": 85},
            {"code": "rock", "name": "Rock", "bpm": 120},
            {"code": "soul", "name": "Soul", "bpm": 103},
            {"code": "soundtracks", "name": "Soundtracks", "bpm": 120},
            {"code": "world", "name": "World", "bpm": 141},
        ]
        return JsonResponse(genres, safe=False)

    except Exception as e:
        print(e)
        return JsonResponse({"message": str(e)}, status=500)


def get_piano_sample(
    request: HttpResponse, sample_name: str
) -> JsonResponse | FileResponse:
    """
    Returns a specific piano sample file.

    Parameters:
    - request (HttpResponse): The incoming HTTP request.
    - sample_name (str): The name of the sample file.

    Returns:
    - JsonResponse: Error messages or success status.
    - FileResponse: The requested sample file.
    """
    if request.method != "GET":
        return JsonResponse({"message": "Wrong method."}, status=405)

    parent_path = Path(__file__).resolve().parent.parent
    sample_path = os.path.join(parent_path, "samples", sample_name)

    print(sample_path)

    if not os.path.exists(sample_path):
        return JsonResponse({"message": f"Sample {sample_name} not found."}, status=404)

    try:
        return FileResponse(open(sample_path, "rb"), content_type="audio/mp3")
    except Exception as e:
        return JsonResponse({"message": str(e)}, status=500)
