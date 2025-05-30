# 🎙️ Accent Detection from Audio

A **Streamlit-based web application** that detects the **accent** of spoken audio using a custom-trained deep learning model and OpenAI’s Whisper for transcription. Supports both audio file uploads and YouTube URLs.

## 📽️ Demo

[![Watch the demo](https://img.shields.io/badge/Watch-Demo-blue)](https://github.com/vijaya-bhargavi/EnAccentDetection/blob/main/Demo.mp4)

<!-- 🔁 Replace the above link with your actual demo video URL -->

---

## 🧠 About the Project

This project classifies **spoken accents** from short audio clips using a combination of transcription (via Whisper) and a deep learning model trained on audio features.

### 🔍 Model Training

* **Preprocessing**:

  * Audio is converted to mono and resampled to 22050 Hz
  * Extracted **MFCC (Mel-Frequency Cepstral Coefficients)** features (40 coefficients)
  * Audio segments were padded or trimmed to 3 seconds for uniformity

* **Model Architecture**:

  * A **Convolutional Neural Network (CNN)** consisting of:

    * Conv2D layers for feature extraction
    * MaxPooling and Dropout layers for downsampling and regularization
    * Dense layers leading to a softmax output for multi-class classification

* **Training Details**:

  * Optimizer: `Adam`
  * Loss Function: `Categorical Crossentropy`
  * Trained for **30 epochs**
  * Evaluation via validation accuracy and loss

---

## 🚀 Features

* 📤 Upload `.mp3` or `.wav` audio files or enter a YouTube URL
* 🔍 Transcribes speech using **OpenAI’s Whisper**
* 🧠 Predicts **accents per segment** of speech
* 📊 Summarizes predictions and visualizes results with bar charts
* 🔁 Supports dynamic segment-wise predictions for varying accents

---

## 🛠️ Tech Stack

| Component          | Library/Tool            |
| ------------------ | ----------------------- |
| Frontend           | Streamlit               |
| Audio Processing   | librosa, yt-dlp, FFmpeg |
| Speech Recognition | Whisper by OpenAI       |
| ML/DL Model        | TensorFlow / Keras      |
| Visualization      | Matplotlib              |
| Utilities          | psutil, gc, pickle      |

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
https://github.com/vijaya-bhargavi/EnAccentDetection.git

cd EnAccentDetection
```

### 2. Install Python Dependencies

> 📄 All required packages are listed in the included `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. FFmpeg Setup 🔧

> 📌 **Note**: FFmpeg is **not included** in the repository due to upload restrictions.

#### Steps:

* Download FFmpeg from: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
* Extract it to a location on your system (e.g., `C:/ffmpeg/ffmpeg/bin`)
* In `app.py`, update this line to your local FFmpeg path:

```python
FFMPEG_PATH = r"C:/ffmpeg/ffmpeg/bin"
```

✅ Make sure the folder contains `ffmpeg.exe` and is added to your system PATH if needed.

### 4. Add Required Model Files

* ✅ Trained model: `best_accent_model.h5`
* ✅ Label encoder: `model_checkpoint/label_encoder.pkl`

These files must be placed in the root/project directory as shown below.

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
accent-detection/
│
├── app.py                          # Streamlit app
├── best_accent_model.h5           # Trained CNN model (user-provided)
├── model_checkpoint/
│   └── label_encoder.pkl          # Pickled LabelEncoder for class decoding
├── requirements.txt               # Python dependencies
└── README.md                      # Project description (this file)
```

---
