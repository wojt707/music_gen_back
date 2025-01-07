import torch.nn as nn
import torch
import music21
import os
from pathlib import Path
import json
import tempfile
import numpy as np


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout=0.2,
    ):
        super(LSTMGenerator, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        word_embed = self.word_embedding(x)  # (batch_size, seq_length, embed_size)

        output, _ = self.lstm(word_embed)
        output = self.fc(output)
        return output[:, -1, :]


def generate_sequence(
    model,
    seed_sequence,
    word_to_idx,
    idx_to_word,
    seq_length,
    length=100,
    temperature=1.0,
):
    """
    Generates sequences using a trained LSTM model.

    Parameters:
    - model: Trained LSTM model.
    - seed_sequence: Initial sequence to start generation.
    - word_to_idx: Dictionary mapping words to indices.
    - idx_to_word: Dictionary mapping indices to words.
    - seq_length: Length of input sequences expected by the model.
    - length: Number of words to generate.
    - temperature: Controls the randomness of predictions (higher = more random).

    Returns:
        List of generated sequence of words.
    """
    # Ensure all words are in the vocabulary
    seed_sequence = [word for word in seed_sequence if word in word_to_idx]

    # Ensure the model is on the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Prepare the starting sequence with padding
    pad_idx = word_to_idx["PAD"]
    start_tokens = [word_to_idx.get(word, pad_idx) for word in seed_sequence]

    if len(start_tokens) < seq_length:
        start_tokens = [pad_idx] * (seq_length - len(start_tokens)) + start_tokens
    else:
        start_tokens = start_tokens[-seq_length:]

    generated_sequence = seed_sequence[:]
    current_sequence = torch.LongTensor([start_tokens]).to(device)

    for _ in range(length - len(seed_sequence)):
        with torch.no_grad():
            output = model(current_sequence)

            logits = output / temperature

            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            next_token_idx = np.random.choice(len(probs), p=probs)

            # Avoid generating "PAD" token
            while next_token_idx == pad_idx:
                next_token_idx = np.random.choice(len(probs), p=probs)

            generated_sequence.append(idx_to_word[str(next_token_idx)])

        # Update the current sequence for the next prediction
        next_input = current_sequence.squeeze(0).tolist()[1:] + [next_token_idx]
        current_sequence = torch.LongTensor([next_input]).to(device)

    return generated_sequence


def create_midi_from_sequence(word_sequence: list, bpm: int, output_path: str):

    midi_stream = music21.stream.Stream()

    bpm = music21.tempo.MetronomeMark(number=max(1, min(bpm, 512)))
    midi_stream.append(bpm)

    current_offset = 0  # Tracks the end of current note or pause in quarter lengths
    last_event_duration = 0
    longest_note_offset = 0  # Tracks the end of note which lasts the longest before pause in quarter lengths

    for word in word_sequence:
        if "PAUSE" in word:
            # Handle pauses
            _, duration_str = word.split("_")
            duration_qlen = music21.duration.Duration(duration_str).quarterLength
            current_offset = longest_note_offset + duration_qlen
            last_event_duration = duration_qlen
            longest_note_offset = 0

        else:
            # Handle notes
            pitch_str, dur_str, rel_start_str = word.split("_")
            pitch = music21.pitch.Pitch(pitch_str)
            duration_qlen = music21.duration.Duration(dur_str).quarterLength
            rel_start_time = music21.duration.Duration(rel_start_str).quarterLength

            # Adjust the note's start time
            current_offset += rel_start_time - last_event_duration

            note = music21.note.Note(pitch)
            note.quarterLength = duration_qlen

            # Add the note to the stream
            midi_stream.insert(current_offset, note)
            current_offset += duration_qlen
            last_event_duration = duration_qlen
            longest_note_offset = max(longest_note_offset, current_offset)

    midi_stream.write("midi", fp=output_path)


def generate_midi_file(genre: str, bpm: int, length: int, temperature: float) -> str:
    """
    Generates a MIDI file based on the given parameters.

    Args:
        genre (str): The genre of the music.
        bpm (int): The tempo in beats per minute.
        length (int): The length of the piece in words (tokens).

    Returns:
        str: The path to the generated MIDI file.
    """
    print(
        f"Starting generation for genre = {genre}, bpm = {bpm}, length = {length}, temperature = {temperature}"
    )

    parent_path = Path(__file__).resolve().parent.parent
    models_path = os.path.join(parent_path, "models")
    midis_path = os.path.join(parent_path, "midis")

    with open(os.path.join(models_path, "word_to_idx.json"), "r") as f:
        word_to_idx = json.load(f)
    with open(os.path.join(models_path, "idx_to_word.json"), "r") as f:
        idx_to_word = json.load(f)

    model_path = os.path.join(models_path, f"{genre}.pt")
    if not os.path.exists(model_path):
        raise Exception(f"Model for genre '{genre}' does not exist.")

    model = LSTMGenerator(len(word_to_idx), 128, 256, 2)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=True,
        )
    )

    # TODO implement various seed sequences
    seed_sequence = [
        "PAUSE_16th",
        "D4_half_zero",
        "C4_eighth_half",
        "A3_whole_half",
        "G3_half_whole",
    ]
    generated = generate_sequence(
        model, seed_sequence, word_to_idx, idx_to_word, seq_length=64, length=length
    )

    # print("Generated Sequence:", " ".join(generated))
    os.makedirs(midis_path, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".mid", dir=midis_path
    ) as temp_file:
        create_midi_from_sequence(generated, bpm, temp_file.name)
        return temp_file.name

    return ""
