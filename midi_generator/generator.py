import torch.nn as nn
import torch
import music21
import os
from pathlib import Path
import json


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
    padding_token="PAD",
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
    - padding_token: Token used for padding.

    Returns:
    - generated_sequence: The generated sequence of words without "PAD".
    """
    # Ensure all words are in the vocabulary
    seed_sequence = [word for word in seed_sequence if word in word_to_idx]

    # Ensure the model is on the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Prepare the starting sequence with padding
    pad_idx = word_to_idx[padding_token]
    start_tokens = [word_to_idx.get(word, pad_idx) for word in seed_sequence]

    if len(start_tokens) < seq_length:
        start_tokens = [pad_idx] * (seq_length - len(start_tokens)) + start_tokens
    else:
        start_tokens = start_tokens[-seq_length:]

    generated_sequence = seed_sequence[:]
    current_sequence = torch.LongTensor([start_tokens]).to(device)

    # Generate tokens
    for _ in range(length - len(seed_sequence)):
        with torch.no_grad():
            output = model(current_sequence)
            # Sort the predictions to find the top two most likely tokens
            probs = torch.softmax(output, dim=1)
            top2 = torch.topk(probs, k=5, dim=1)
            next_token_idx = top2.indices[0, 0].item()

            # If the most likely token is "PAD", use the second most likely token
            if next_token_idx == pad_idx:
                next_token_idx = top2.indices[0, 1].item()

            next_token = idx_to_word[str(next_token_idx)]

            generated_sequence.append(next_token)

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


def generate_midi_file(genre: str, bpm: int, length: int, randomness: float) -> str:
    """
    Generates a MIDI file based on the given parameters.

    Args:
        genre (str): The genre of the music (for now, it just tags the file).
        bpm (int): The tempo in beats per minute.
        length (int): The length of the piece in words (tokens).

    Returns:
        str: The path to the generated MIDI file.
    """
    print(
        f"Starting generation for genre = {genre}, bpm = {bpm}, length = {length}, randomness = {randomness}"
    )
    parent_path = Path(__file__).resolve().parent.parent
    models_path = os.path.join(parent_path, "models")
    midis_path = os.path.join(parent_path, "midis")

    with open(os.path.join(models_path, "word_to_idx.json"), "r") as f:
        word_to_idx = json.load(f)
    with open(os.path.join(models_path, "idx_to_word.json"), "r") as f:
        idx_to_word = json.load(f)

    vocab_size = len(word_to_idx)

    embed_size = 128
    hidden_size = 256
    num_layers = 2
    seq_length = 64

    model = LSTMGenerator(vocab_size, embed_size, hidden_size, num_layers)

    model_path = os.path.join(models_path, f"{genre}.pt")

    if not os.path.exists(model_path):
        raise Exception(f"Model {genre} does not exist.")

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=True,
        )
    )

    # TODO implement various seed sequences
    seed_sequence = "PAUSE_16th D4_half_zero D4_eighth_half D4_eighth_eighth A3_eighth_eighth".split(
        " "
    )

    generated = generate_sequence(
        model, seed_sequence, word_to_idx, idx_to_word, seq_length, length
    )

    print("Generated Sequence:", " ".join(generated))

    os.makedirs(midis_path, exist_ok=True)
    midi_file_path = os.path.join(midis_path, f"{genre}.mid")

    create_midi_from_sequence(generated, bpm, midi_file_path)
    # create_midi_from_sequence(seed_sequence, bpm, os.path.join(midis_path, "sth.mid"))

    return midi_file_path
