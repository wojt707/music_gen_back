# MidiForge — Backend

Flask backend that loads pretrained LSTM genre models and serves generated MIDI and sample audio.

## Purpose
Expose simple API endpoints used by the frontend to:
- list genres
- generate MIDI files
- serve piano sample audio files

## Tech
- Python, Flask
- PyTorch (model loading), music21 (MIDI creation)

## Quick start (local)
1. Create virtualenv and install:
   ```
   python -m venv .venv
   source .venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Run:
   ```
   flask --app main run
   ```
3. Or with Docker:
   ```
   docker build -t music_gen_back .
   docker run -e PORT=5000 -p 5000:5000 music_gen_back
   ```

## API
- GET /api/genres — list genres and default BPMs
- POST /api/generate — body: {"genre","bpm","length","randomness"} → returns .mid file
- GET /api/samples/<sample_name> — returns mp3 sample

## Files
- models/ — pretrained .pt models and vocab maps
- seeds/ — seed token files used to start generation
- generator.py — generation logic
- main.py — Flask routes

## Env
- PORT (optional, used by Docker / gunicorn)

## Notes
- Generated MIDI is written to /tmp and removed after serving.
- randomness is mapped to sampling temperature; expected roughly in [-100,100].

## License
See [LICENSE](LICENSE) file.
