import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import pretty_midi
import librosa
import pandas as pd
from piano_transcription_inference import PianoTranscription  # Ajusta el import según tu instalación

sample_rate = 16000

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def downsample_matrix(matrix, original_fs, target_fs):
    factor = original_fs // target_fs
    new_len = matrix.shape[0] // factor
    return matrix[:new_len * factor].reshape(new_len, factor, -1).max(axis=1)

def get_pianoroll_from_mp3(mp3_path):
    audio, _ = librosa.load(mp3_path, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device='cuda')
    midi_path = "temp.mid"
    transcriptor.transcribe(audio, midi_path)
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    fs = 100
    piano_roll = midi_data.get_piano_roll(fs=5)[21:109].T
    piano_roll = piano_roll / 127
    time_steps = piano_roll.shape[0]

    onsets = np.zeros_like(piano_roll)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitch = note.pitch - 21
            onset_frame = int(note.start * fs)
            if 0 <= pitch < 88 and onset_frame < time_steps:
                onsets[onset_frame, pitch] = 1.0

    pr_tensor = torch.tensor(piano_roll.T).unsqueeze(0).unsqueeze(1).cuda().float()
    on_tensor = torch.tensor(onsets.T).unsqueeze(0).unsqueeze(1).cuda().float()
    out_tensor = torch.cat([pr_tensor, on_tensor], dim=1)
    print(f"piano_roll shape: {out_tensor.shape}")
    return out_tensor.transpose(2, 3)

def save_to_excel(piano_roll_bin, onsets_bin, pr_mp3, on_mp3, path='pianoroll_comparison.xlsx'):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(piano_roll_bin).to_excel(writer, sheet_name="piano_roll_bin", index=False)
        pd.DataFrame(onsets_bin).to_excel(writer, sheet_name="onsets_bin", index=False)
        pd.DataFrame(pr_mp3).to_excel(writer, sheet_name="pr_mp3", index=False)
        pd.DataFrame(on_mp3).to_excel(writer, sheet_name="on_mp3", index=False)
    print(f"Excel guardado en: {path}")

def main():
    path_base = "../videos_download/pianoroll5/Absil J.Humoresque Op 126 No 3"
    piano_roll_bin = load_pickle(f"{path_base}.bin")
    onsets_bin = load_pickle(f"{path_base}_onset.bin")

    print("From bin:")
    print("piano_roll_bin shape:", piano_roll_bin.shape)
    print("onsets_bin shape:", onsets_bin.shape)

    mp3_path = "yt_audio.mp3"
    tensor_from_mp3 = get_pianoroll_from_mp3(mp3_path)

    pr_mp3 = tensor_from_mp3[0, 0].cpu().numpy()
    on_mp3 = tensor_from_mp3[0, 1].cpu().numpy()

    same_pr = np.allclose(pr_mp3, piano_roll_bin, atol=1e-3)
    same_on = np.allclose(on_mp3, onsets_bin, atol=1e-3)

    # visualize
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.title("Piano Roll - Binary")
    plt.imshow(piano_roll_bin.T, aspect='auto', cmap='hot')
    plt.colorbar()
    plt.ylabel("Pitch")
    plt.xlabel("Time")

    plt.subplot(2, 1, 2)
    plt.title("Piano Roll - MP3")
    plt.imshow(pr_mp3.T, aspect='auto', cmap='hot')
    plt.colorbar()
    plt.ylabel("Pitch")
    plt.xlabel("Time")

    plt.tight_layout()
    plt.show()

    print("Comparison results:")
    print("Same piano roll:", same_pr)
    print("Same onsets:", same_on)

    if not same_pr:
        print("Max diff piano roll:", np.max(np.abs(pr_mp3 - piano_roll_bin)))
    if not same_on:
        print("Max diff onsets:", np.max(np.abs(on_mp3 - onsets_bin)))

    save_to_excel(piano_roll_bin, onsets_bin, pr_mp3, on_mp3)

if __name__ == '__main__':
    main()
