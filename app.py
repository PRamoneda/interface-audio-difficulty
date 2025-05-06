import gradio as gr
from get_difficulty import predict_difficulty
import tempfile
import os
from pydub import AudioSegment



import yt_dlp

def download_youtube_audio(url):
    output_path = "yt_audio.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return "yt_audio.mp3"


def process_input(input_audio, youtube_url):
    if youtube_url:
        audio_path = download_youtube_audio(youtube_url)
    elif input_audio:
        audio_path = input_audio  # Gradio devuelve una ruta de archivo si type="filepath"
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_segment = AudioSegment.from_file(audio_path)
        audio_segment.export(temp_audio.name, format="mp3")
        audio_path = temp_audio.name
    else:
        return "No audio provided.", None, None

    model_cqt = "audio_midi_cqt5_ps_v5"
    model_pr = "audio_midi_pianoroll_ps_5_v4"
    model_multi = "audio_midi_multi_ps_v5"

    diff_cqt = predict_difficulty(audio_path, model_name=model_cqt, rep="cqt5")
    diff_pr = predict_difficulty(audio_path, model_name=model_pr, rep="pianoroll5")
    diff_multi = predict_difficulty(audio_path, model_name=model_multi, rep="multimodal5")

    # Assumes predict_difficulty generates 'temp.mid'
    midi_path = "temp.mid"
    if not os.path.exists(midi_path):
        return "MIDI not generated.", None, None

    return (
        f"CQT difficulty: {diff_cqt}\n"
        f"Pianoroll difficulty: {diff_pr}\n"
        f"Multimodal difficulty: {diff_multi}",
        midi_path,
        midi_path
    )


demo = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(label="Upload MP3", type="filepath"),
        gr.Textbox(label="YouTube URL")
    ],
    outputs=[
        gr.Textbox(label="Difficulty predictions"),
        gr.File(label="Generated MIDI"),
        gr.Audio(label="MIDI Playback", type="filepath")
    ],
    title="Music Difficulty Estimator",
    description="Upload an MP3 or provide a YouTube URL. It predicts difficulty and generates a MIDI file you can listen to."
)

if __name__ == "__main__":
    demo.launch()
