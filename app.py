import gradio as gr
from get_difficulty import predict_difficulty
import tempfile
import os
from pydub import AudioSegment
import yt_dlp
import mimetypes  # NEW

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

def convert_to_mp3(input_path):
    audio = AudioSegment.from_file(input_path)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio.export(temp_audio.name, format="mp3")
    return temp_audio.name

def process_input(input_file, youtube_url):
    audio_path = None
    mp3_path = None

    if youtube_url:
        audio_path = download_youtube_audio(youtube_url)
        mp3_path = audio_path
    elif input_file:
        mime_type, _ = mimetypes.guess_type(input_file)
        if mime_type and mime_type.startswith("video/"):  # NEW: MP4 support
            audio_path = convert_to_mp3(input_file)
            mp3_path = audio_path
        else:
            audio_path = convert_to_mp3(input_file)
            mp3_path = audio_path
    else:
        return "No audio or video provided.", None, None, None

    model_cqt = "audio_midi_cqt5_ps_v5"
    model_pr = "audio_midi_pianoroll_ps_5_v4"
    model_multi = "audio_midi_multi_ps_v5"

    diff_cqt = predict_difficulty(audio_path, model_name=model_cqt, rep="cqt5")
    diff_pr = predict_difficulty(audio_path, model_name=model_pr, rep="pianoroll5")
    diff_multi = predict_difficulty(audio_path, model_name=model_multi, rep="multimodal5")

    midi_path = "temp.mid"
    if not os.path.exists(midi_path):
        return "MIDI not generated.", None, None, None

    difficulty_text = (
        f"CQT difficulty: {diff_cqt}\n"
        f"Pianoroll difficulty: {diff_pr}\n"
        f"Multimodal difficulty: {diff_multi}"
    )

    return difficulty_text, midi_path, midi_path, mp3_path  # NEW: mp3_path added

demo = gr.Interface(
    fn=process_input,
    inputs=[
        gr.File(label="Upload MP3 or MP4", type="filepath"),  # NEW
        gr.Textbox(label="YouTube URL")
    ],
    outputs=[
        gr.Textbox(label="Difficulty predictions"),
        gr.File(label="Generated MIDI"),
        gr.Audio(label="MIDI Playback", type="filepath"),
        gr.Audio(label="Extracted MP3 Preview", type="filepath")  # NEW
    ],
    title="Music Difficulty Estimator",
    description="Upload an MP3, MP4, or provide a YouTube URL. It extracts audio, predicts difficulty, and generates a MIDI file."
)

if __name__ == "__main__":
    demo.launch()