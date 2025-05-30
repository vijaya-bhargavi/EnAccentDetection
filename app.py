import os
import gc
import shutil
import pickle
import warnings
from collections import Counter

import numpy as np
import streamlit as st
import librosa
import whisper
import yt_dlp
import matplotlib.pyplot as plt
import psutil
from tensorflow.keras.models import load_model

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Streamlit file watcher warnings (Windows fix)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Streamlit page settings
st.set_page_config(page_title="Accent Detection")
st.title("Accent Detection from Audio or YouTube")

# Display memory usage before model loading
gc.collect()
process = psutil.Process(os.getpid())
print(f"Memory used before Whisper load: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Constants
MODEL_PATH = "best_accent_model.h5"
ENCODER_PATH = "model_checkpoint/label_encoder.pkl"
FFMPEG_PATH = r"./ffmpeg-2025-05-29-git-75960ac270-full_build/bin"

os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Suppress warnings
warnings.filterwarnings("ignore")

# Load model and label encoder
@st.cache_resource
def load_model_assets():
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_assets()

# Input method selection
input_method = st.radio("Choose Input Type", ["Upload Audio File", "YouTube URL"])
audio_file_path = None
temp_dir = os.path.join(os.getcwd(), "temp_audio")
os.makedirs(temp_dir, exist_ok=True)

# Upload audio file

import re
if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    if uploaded_file:
       
        original_filename = uploaded_file.name
        safe_filename = re.sub(r'[<>:"/\\|?*\']', '_', original_filename)       
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True) 
        audio_file_path = os.path.join(temp_dir, safe_filename)
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info(f"Audio file uploaded successfully as: {safe_filename}")


# YouTube input
if input_method == "YouTube URL":
    url = st.text_input("Enter YouTube video URL")
    if url:
        download_status = st.empty()
        download_status.info("Downloading and processing audio from YouTube...")

        audio_file_template = os.path.join(temp_dir, "audio.%(ext)s")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_file_template,
            'quiet': True,
            'ffmpeg_location': FFMPEG_PATH,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            for f in os.listdir(temp_dir):
                if f.endswith(".wav"):
                    audio_file_path = os.path.join(temp_dir, f)
                    break
            download_status.success("Audio download and conversion completed.")
        except Exception as e:
            download_status.error(f"Download failed: {e}")
            shutil.rmtree(temp_dir)
            st.stop()

# Prediction function
def predict_accent(audio_path):
    st.subheader("Processing Audio")
    status_placeholder = st.empty()
    status_placeholder.info("Loading Whisper model and transcribing speech...")

    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(audio_path)
    segments = result.get("segments", [])

    status_placeholder.success(f"Transcription complete. {len(segments)} segments found.")

    predictions = []
    progress_bar = st.progress(0)
    status_placeholder.info("Predicting accent for each segment...")

    for idx, seg in enumerate(segments):
        try:
            start, end = seg['start'], seg['end']
            y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=end - start)
            y = np.pad(y, (0, max(0, sr * 3 - len(y))))[:sr * 3]
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
            mfcc = np.expand_dims(mfcc, axis=(0, -1))
            pred = model.predict(mfcc, verbose=0)
            pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]
            predictions.append((start, end, pred_label, np.max(pred)))
        except:
            predictions.append((seg['start'], seg['end'], "error", 0.0))

        progress_bar.progress((idx + 1) / len(segments))

    progress_bar.empty()
    status_placeholder.success("Accent prediction completed.")
    return predictions

# Display results
def display_results(predictions):
    st.subheader("Summary of Predictions")

    valid_preds = [p for p in predictions if p[2] != "error"]
    pred_labels = [p[2] for p in valid_preds]

    st.write(f"Total segments analyzed: {len(valid_preds)}")

    if pred_labels:
        counts = Counter(pred_labels)
        main_label, main_count = counts.most_common(1)[0]
        st.info(f"Detected most frequent accent: {main_label} (occurrences: {main_count})")

        # Plot bar chart
        top = counts.most_common(5)
        labels, values = zip(*top)
        fig, ax = plt.subplots()
        ax.barh(labels, values, color="steelblue")
        ax.set_xlabel("Count")
        ax.set_title("Top Accent Predictions")
        st.pyplot(fig)
    else:
        st.warning("No valid predictions were generated.")

# Run pipeline
if audio_file_path:
    with st.spinner("Running accent detection pipeline..."):
        preds = predict_accent(audio_file_path)
    display_results(preds)
    shutil.rmtree(temp_dir)