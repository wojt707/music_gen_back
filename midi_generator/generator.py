import torch.nn as nn
import torch
import music21
import os
from pathlib import Path
import json
import tempfile
import numpy as np
import random


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        """
        Initializes the LSTM Generator model.

        Parameters:
        - vocab_size (int): Size of the vocabulary (number of unique tokens).
        - embed_size (int): Size of the embedding vector for each token.
        - hidden_size (int): Number of hidden units in the LSTM.
        - num_layers (int): Number of layers in the LSTM.
        - dropout (float): Dropout rate applied to the LSTM layers.
        """
        super(LSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, prev_state: tuple) -> tuple:
        """
        Forward pass for generating predictions.

        Parameters:
        - x (torch.Tensor): Input tensor of word indices.
        - prev_state (tuple): The previous hidden and cell states of the LSTM.

        Returns:
        - tuple: Logits from the fully connected layer and the updated states.
        """
        word_embed = self.word_embedding(x)  # (batch_size, seq_length, embed_size)

        output, state = self.lstm(
            word_embed, prev_state
        )  # (batch_size, seq_length, lstm_size)
        logits = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return logits, state

    def init_state(self, batch_size: int) -> tuple:
        """
        Initializes the hidden and cell states of the LSTM.

        Parameters:
        - batch_size (int): The batch size for generating sequences.

        Returns:
        - tuple: Initialized hidden and cell states.
        """
        return (
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=self.device
            ),
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=self.device
            ),
        )


def map_randomness_to_temperature(randomness: float) -> float:
    """
    Maps the randomness parameter to a temperature value.

    Parameters:
    - randomness (float): The randomness factor between -100 and 100.

    Returns:
    - float: Corresponding temperature value between 0.1 and 10.
    """
    randomness = min(100.0, max(-100.0, randomness))

    if randomness >= 0.0:
        temperature = 9.0 * randomness / 100.0 + 1.0
    else:
        temperature = 0.9 * (randomness / 100.0 + 1.0) + 0.1

    return temperature


def generate_sequence(
    model: nn.Module,
    seed_sequence: list,
    word_to_idx: dict,
    idx_to_word: dict,
    seq_length: int,
    length: int = 100,
    temperature: float = 1.0,
) -> list:
    """
    Generates a sequence using a trained LSTM model.

    Parameters:
    - model (nn.Module): Trained LSTM model.
    - seed_sequence (list): Initial sequence to start generation.
    - word_to_idx (dict): Mapping of words to indices.
    - idx_to_word (dict): Mapping of indices to words.
    - seq_length (int): The length of input sequences expected by the model.
    - length (int): The number of tokens to generate.
    - temperature (float): Controls randomness of predictions (higher = more random).

    Returns:
    - list: Generated sequence of words.
    """
    seed_sequence = [word for word in seed_sequence if word in word_to_idx]

    model.to(model.device)
    model.eval()

    # Prepare the starting sequence with padding
    pad_idx = word_to_idx["PAD"]
    start_tokens = [word_to_idx.get(word, pad_idx) for word in seed_sequence]

    if len(start_tokens) < seq_length:
        start_tokens = [pad_idx] * (seq_length - len(start_tokens)) + start_tokens
    else:
        start_tokens = start_tokens[-seq_length:]

    generated_sequence = seed_sequence[:]
    current_sequence = torch.LongTensor([start_tokens]).to(model.device)

    state_h, state_c = model.init_state(batch_size=1)

    for _ in range(length - len(seed_sequence)):
        with torch.no_grad():
            logits, (state_h, state_c) = model(current_sequence, (state_h, state_c))

            # Focus on the last timestep's output and apply temperature scaling
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            next_token_idx = np.random.choice(len(probs), p=probs)

            # Avoid generating "PAD" token
            while next_token_idx == pad_idx:
                next_token_idx = np.random.choice(len(probs), p=probs)

            generated_sequence.append(idx_to_word[str(next_token_idx)])

        next_input = current_sequence.squeeze(0).tolist()[1:] + [next_token_idx]
        current_sequence = torch.LongTensor([next_input]).to(model.device)

    return generated_sequence


def create_midi_from_sequence(word_sequence: list, bpm: int, output_path: str):
    """
    Converts a sequence of words (representing music events) into a MIDI file.

    Parameters:
    - word_sequence (list): The sequence of generated words (representing music events).
    - bpm (int): The tempo (beats per minute).
    - output_path (str): The path where the generated MIDI file will be saved.
    """
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

            midi_stream.insert(current_offset, note)
            current_offset += duration_qlen
            last_event_duration = duration_qlen
            longest_note_offset = max(longest_note_offset, current_offset)

    midi_stream.write("midi", fp=output_path)


def get_random_seed(idx_to_word: dict, genre: str, seed_folder: str) -> list:
    """
    Fetches a random seed from pre-generated seeds for a specific genre.

    Parameters:
    - idx_to_word (dict): Mapping of indices to words.
    - genre (str): The genre of the seed.
    - seed_folder (str): Path to the folder containing seed files.

    Returns:
    - list: The seed sequence as a list of words.
    """
    seed_file = os.path.join(seed_folder, f"{genre}_seeds.txt")
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"Seed file for genre {genre} not found.")

    with open(seed_file, "r") as f:
        seeds = f.readlines()

    if not seeds:
        raise ValueError(f"No seeds found in the seed file for genre {genre}.")

    token_seed = random.choice(seeds).strip().split()
    word_seed = [idx_to_word[token] for token in token_seed]

    print(token_seed)
    print(word_seed)
    return word_seed


def generate_midi_file(genre: str, bpm: int, length: int, randomness: float) -> str:
    """
    Generates a MIDI file based on the given parameters.

    Parameters:
    - genre (str): The genre of the music.
    - bpm (int): The tempo (beats per minute).
    - length (int): The length of the generated sequence (in words).
    - randomness (float): Controls the randomness of the generation.

    Returns:
    - str: Path to the generated MIDI file.
    """
    print(
        f"Starting generation for genre = {genre}, bpm = {bpm}, length = {length}, randomness = {randomness}"
    )

    temperature = map_randomness_to_temperature(randomness)

    parent_path = Path(__file__).resolve().parent.parent
    models_path = os.path.join(parent_path, "models")
    midis_path = os.path.join(parent_path, "midis")
    seeds_path = os.path.join(parent_path, "seeds")

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

    seed_sequence = get_random_seed(idx_to_word, genre, seeds_path)

    generated = generate_sequence(
        model,
        seed_sequence,
        word_to_idx,
        idx_to_word,
        seq_length=64,
        length=length,
        temperature=temperature,
    )

    os.makedirs(midis_path, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".mid", dir=midis_path
    ) as temp_file:
        create_midi_from_sequence(generated, bpm, temp_file.name)
        return temp_file.name

    return ""
