import os
from statistics import mean

import torch
from torch import nn
import numpy as np
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import pretty_midi
import soundfile as sf
from utils import prediction2label
from model import AudioModel
from scipy.signal import resample


def downsample_log_cqt(cqt_matrix, target_fs=5):
    original_fs = 44100 / 160
    ratio = original_fs / target_fs
    downsampled = resample(cqt_matrix, int(cqt_matrix.shape[0] / ratio), axis=0)
    return downsampled

def downsample_matrix(mat, original_fs, target_fs):
    ratio = original_fs / target_fs
    return resample(mat, int(mat.shape[0] / ratio), axis=0)

def get_cqt_from_mp3(mp3_path):
    sample_rate = 44100
    hop_length = 160
    y, sr = librosa.load(mp3_path, sr=sample_rate, mono=True)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=88, bins_per_octave=12)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    log_cqt = log_cqt.T  # shape (T, 88)
    log_cqt = downsample_log_cqt(log_cqt, target_fs=5)
    cqt_tensor = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    return cqt_tensor.transpose(-1, -2)

def get_pianoroll_from_mp3(mp3_path):
    audio, _ = load_audio(mp3_path, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device='cuda')
    midi_path = "temp.mid"
    transcriptor.transcribe(audio, midi_path)
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # Create pianoroll and onset matrix
    fs = 100  # original frames per second
    piano_roll = midi_data.get_piano_roll(fs=fs)[21:109].T  # shape: (T, 88)
    piano_roll = np.clip(piano_roll / 127, 0, 1)
    time_steps = piano_roll.shape[0]

    onsets = np.zeros_like(piano_roll)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitch = note.pitch - 21
            onset_frame = int(note.start * fs)
            if 0 <= pitch < 88 and onset_frame < time_steps:
                onsets[onset_frame, pitch] = 1.0

    piano_roll = downsample_matrix(piano_roll, original_fs=fs, target_fs=5)
    onsets = downsample_matrix(onsets, original_fs=fs, target_fs=5)

    pr_tensor = torch.tensor(piano_roll.T).unsqueeze(0).unsqueeze(1).cuda().float()
    on_tensor = torch.tensor(onsets.T).unsqueeze(0).unsqueeze(1).cuda().float()
    out_tensor = torch.cat([pr_tensor, on_tensor], dim=1)
    return out_tensor

def predict_difficulty(mp3_path, model_name, rep):
    if "only_cqt" in rep:
        only_cqt, only_pr = True, False
        rep_clean = "multimodal5"
    elif "only_pr" in rep:
        only_cqt, only_pr = False, True
        rep_clean = "multimodal5"
    else:
        only_cqt = only_pr = False
        rep_clean = rep

    model = AudioModel(num_classes=11, rep=rep_clean, modality_dropout=False, only_cqt=only_cqt, only_pr=only_pr)
    checkpoint = [torch.load(f"models/{model_name}/checkpoint_{i}.pth", map_location="cuda", weights_only=False)
                  for i in range(5)]


    if rep == "cqt5":
        inp_data = get_cqt_from_mp3(mp3_path)
    elif rep == "pianoroll5":
        inp_data = get_pianoroll_from_mp3(mp3_path)
    elif rep_clean == "multimodal5":
        x1 = get_pianoroll_from_mp3(mp3_path)
        x2 = get_cqt_from_mp3(mp3_path)
        inp_data = [x1, x2]
    else:
        raise ValueError(f"Representation {rep} not supported")

    preds = []
    for cheks in checkpoint:
        model.load_state_dict(cheks["model_state_dict"])
        model = model.cuda().eval()
        with torch.inference_mode():
            logits = model(inp_data, None)
            pred = prediction2label(logits).item()
            preds.append(pred)

    return mean(preds)


if __name__ == "__main__":
    mp3_path = "chopin.mp3"
    model_name = "audio_midi_multi_ps_v5"

    pred_cqt = predict_difficulty(mp3_path, model_name="audio_midi_cqt5_ps_v5", rep="cqt5")
    print(f"Predicción dificultad CQT: {pred_cqt}")

    pred_pr = predict_difficulty(mp3_path, model_name="audio_midi_pianoroll_ps_5_v4", rep="pianoroll5")
    print(f"Predicción dificultad PR: {pred_pr}")

    pred_multi = predict_difficulty(mp3_path, model_name=model_name, rep="multimodal5")
    print(f"Predicción dificultad multimodal: {pred_multi}")