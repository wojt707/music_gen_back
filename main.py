from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

import os
from io import BytesIO

from generator import generate_midi_file

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://midiforge.onrender.com"])


@app.route("/")
def home():
    return "Flask backend is working!"


@app.get("/api/genres")
def get_genres():
    """
    Returns the list of available genres.

    Returns:
    - JsonResponse: A list of genre dictionaries or error messages.
    """
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
        return jsonify(genres)

    except Exception as e:
        print(e)
        return jsonify({"message": str(e), "status": 500}), 500


@app.post("/api/generate")
def generate_midi():
    """
    Handles MIDI file generation requests.

    Returns:
    - JSON response with error or success messages.
    - File response upon successful MIDI generation.
    """
    try:
        # Parse and validate the request data
        data = request.get_json()

        genre = data.get("genre")
        bpm = data.get("bpm")
        length = data.get("length")
        randomness = data.get("randomness")

        print(genre)
        if not genre or not isinstance(genre, str):
            return jsonify({"message": "Invalid or missing genre.", "status": 400}), 400
        try:
            bpm = int(bpm)
            if bpm <= 0:
                raise ValueError
        except Exception:
            return jsonify({"message": f"Invalid BPM: {bpm}", "status": 400}), 400
        try:
            length = int(length)
            if length <= 0:
                raise ValueError
        except Exception:
            return jsonify({"message": f"Invalid length: {length}", "status": 400}), 400
        try:
            randomness = float(randomness)
            if randomness < -100 or randomness > 100:
                raise ValueError
        except Exception:
            return (
                jsonify(
                    {"message": f"Invalid randomness: {randomness}", "status": 400}
                ),
                400,
            )

        # Generate the MIDI file
        midi_path = generate_midi_file(genre, bpm, length, randomness)
        if not os.path.exists(midi_path):
            return jsonify({"message": "MIDI generation failed.", "status": 500})

        with open(midi_path, "rb") as mid:
            midi_data = BytesIO(mid.read())
        os.remove(midi_path)
        midi_data.seek(0)
        return send_file(midi_data, as_attachment=True, download_name=f"{genre}.mid")

    except Exception as e:
        print(f"Error generating MIDI: {e}")
        return jsonify({"message": str(e), "status": 500}), 500


@app.get("/api/samples/<sample_name>")
def get_piano_sample(sample_name):
    """
    Returns a specific piano sample file.

    Returns:
    - File response with the sample file.
    """
    try:
        parent_path = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(parent_path, "samples", sample_name)

        if not os.path.exists(sample_path):
            return (
                jsonify({"message": f"Sample {sample_name} not found.", "status": 404}),
                404,
            )

        return send_file(sample_path, mimetype="audio/mp3")
    except Exception as e:
        return jsonify({"message": str(e), "status": 500}), 500
