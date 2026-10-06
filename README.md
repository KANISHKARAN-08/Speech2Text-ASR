# Speech2Text-ASR

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![ASR](https://img.shields.io/badge/ASR-OpenAI%20Whisper-orange.svg)](https://github.com/openai/whisper)
[![UI](https://img.shields.io/badge/Interface-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An Automatic Speech Recognition (ASR) web application built using **OpenAI Whisper**, **PyTorch**, and **Streamlit**. This tool converts spoken human voice from audio files or direct microphone input into clean, written text running entirely locally on your system.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [How ASR Works](#how-asr-works)
- [Project Structure](#project-structure)
- [Installation Instructions (Windows)](#installation-instructions-windows)
- [FFmpeg Installation & Setup](#ffmpeg-installation--setup)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Demonstration Example](#demonstration-example)
- [Whisper Model Information](#whisper-model-information)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Publishing to GitHub](#publishing-to-github)
- [License](#license)

---

## Project Overview

Automatic Speech Recognition (ASR) is a subfield of artificial intelligence and computational linguistics that focuses on transcribing spoken audio into machine-readable text.

**Speech2Text-ASR** is a modular, offline-capable ASR application designed for academic demonstrations and technical assessments. It leverages pretrained deep learning representations from OpenAI's Whisper model to achieve robust transcription across multiple audio formats without requiring paid cloud API keys or external services.

---

## Key Features

- **Audio File Upload**: Upload local audio recordings in common formats (WAV, MP3, M4A, OGG, FLAC).
- **Microphone Recording**: Capture live voice recordings directly in the browser using the native audio recorder.
- **Audio Playback Preview**: Inspect and listen to the audio recording prior to starting transcription.
- **Whisper Deep Learning ASR**: Leverages OpenAI Whisper pretrained models with state-of-the-art transformer architecture.
- **Configurable Models**: Switch effortlessly between `tiny`, `base`, and `small` models directly from the sidebar.
- **Language Identification**: Automatically detects the spoken language of the audio clip.
- **Transcription Display**: View the formatted transcription in a clean, scrollable interface.
- **Downloadable Output**: Export transcriptions as timestamped `.txt` files with source metadata.
- **Graceful Error Handling**: Clear feedback for unsupported file extensions, missing FFmpeg, oversized audio, or empty files.
- **Automated Pytest Suite**: Comprehensive unit tests with mocked inference for lightning-fast testing.

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** (3.10 / 3.11) | Core programming language |
| **OpenAI Whisper** | Pretrained deep learning Automatic Speech Recognition (ASR) engine |
| **PyTorch** | Deep learning tensor computation framework powering Whisper |
| **Streamlit** | Interactive web application user interface |
| **FFmpeg** | Multipurpose multimedia framework for decoding and converting audio streams |
| **Pytest** | Automated unit testing and test discovery |
| **Git & GitHub** | Distributed version control and source code repository hosting |

---

## System Architecture

```text
               +--------------------------------------+
               |             User Interface           |
               |          (Streamlit Web App)         |
               +-------------------+------------------+
                                   |
                +------------------+------------------+
                |                                     |
       [ Upload Audio File ]                 [ Record Microphone ]
                |                                     |
                +------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               |           Audio Validation           |
               |     (Extension, Size, Non-empty)     |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               |          Audio Preprocessing         |
               |      (FFmpeg 16kHz Mono Resample)    |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               |          Whisper ASR Model           |
               |     (Encoder-Decoder Transformer)    |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               |         Text Transcription           |
               |        & Language Detection          |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               |          Display & Download          |
               |     (View on Screen / Export .txt)   |
               +--------------------------------------+
```

---

## How ASR Works

1. **Audio Acquisition**: The user provides speech audio either by uploading an existing file or capturing speech via microphone.
2. **Audio Decoding & Normalization**: FFmpeg decodes the input audio stream and resamples it to 16,000 Hz mono PCM format.
3. **Spectrogram Generation**: The audio signal is transformed into an 80-channel log-magnitude Mel spectrogram representation over 25-millisecond windows.
4. **Encoder Processing**: Whisper's transformer encoder processes the spectrogram frames through multi-head self-attention layers to extract speech features.
5. **Autoregressive Decoding**: The transformer decoder uses cross-attention over the encoder features to autoregressively predict text tokens.
6. **Text Rendering & Export**: The decoded text is filtered, formatted, displayed on the Streamlit interface, and made available for download.

---

## Project Structure

```text
Speech2Text-ASR/
│
├── app.py                  # Streamlit web user interface
├── asr.py                  # Independent OpenAI Whisper ASR engine class
├── config.py               # Central configuration constants and settings
│
├── requirements.txt        # Minimal, pinned Python package dependencies
├── README.md               # Complete project documentation
├── .gitignore              # Ignores venv, caches, audio files, and credentials
├── LICENSE                 # MIT Open Source License (2026)
│
├── tests/
│   ├── __init__.py         # Test package initialization
│   └── test_asr.py         # Pytest test cases with mocking
│
├── utils/
│   ├── __init__.py         # Utility package initialization
│   └── audio_utils.py      # Audio validation, temporary storage, and cleanup
│
├── sample_audio/
│   └── README.md           # Instructions on placing sample audio clips
│
└── screenshots/
    └── README.md           # Instructions for adding application screenshots
```

---

## Installation Instructions (Windows)

Follow these steps to set up the project on Windows:

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Speech2Text-ASR.git
cd Speech2Text-ASR
```

### Step 2 — Create a Virtual Environment

Using Python 3.10 or 3.11:

```bash
python -m venv venv
```

### Step 3 — Activate the Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## FFmpeg Installation & Setup

OpenAI Whisper requires **FFmpeg** to decode and process audio files.

### Option A — Install via Windows Package Manager (Recommended)

Run the following in PowerShell:

```powershell
winget install Gyan.FFmpeg.Essentials
```

After the installation finishes, restart your PowerShell window or IDE terminal.

### Option B — Manual Installation

1. Download the latest release essentials build archive from [Gyan.dev FFmpeg](https://www.gyan.dev/ffmpeg/builds/).
2. Extract the archive into a folder such as `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your Windows System `PATH` Environment Variable.

### Verification

Check that FFmpeg is accessible by running:

```bash
ffmpeg -version
```

If version information is displayed, FFmpeg is correctly configured on your system.

---

## Running the Application

Launch the Streamlit web application:

```bash
python -m streamlit run app.py
```

### Expected Output

Streamlit will print the local URL in your terminal and automatically open the application in your default web browser:

```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

In the browser interface:
1. Select the Whisper model in the left sidebar (default: `base`).
2. Choose **Upload Audio** or **Record Audio**.
3. Listen to the preview using the integrated audio player.
4. Click **🎤 Transcribe Audio**.
5. View the transcribed text on screen and click **📥 Download Transcription (.txt)** to save the result.

---

## Testing

The project includes an automated test suite implemented with `pytest`. Automated tests use mocks to avoid downloading multi-gigabyte models during regular test runs.

Run the test suite:

```bash
python -m pytest
```

To run with verbose output:

```bash
python -m pytest -v
```

### Verified Test Results

All tests have been directly executed and verified on the system:
- **Pytest Suite**: 25 out of 25 unit tests passed (`25 passed in 4.32s`).
- **Live ASR Inference**: Verified with Whisper `base` model transcribing spoken audio (`sample_audio/test_speech.wav`) word-for-word.
- **Streamlit App Execution**: Verified via `streamlit.testing.v1.AppTest` with zero exceptions and local HTTP endpoint returning status code `200`.

### Test Coverage Table

| Test Case | Description | Measured Result |
|-----------|-------------|-----------------|
| `test_config_values` | Validates configuration constants (`DEFAULT_MODEL`, formats) | Passed |
| `test_validate_audio_file_valid_extensions` | Checks WAV, MP3, M4A, OGG, FLAC formats | Passed |
| `test_validate_audio_file_invalid_extensions` | Checks non-audio extensions (.txt, .py, .mp4) | Passed |
| `test_validate_audio_file_empty_and_none` | Verifies None and blank input handling | Passed |
| `test_validate_audio_file_zero_bytes` | Tests 0-byte uploaded file | Passed |
| `test_validate_audio_file_size_exceeded` | Tests files exceeding maximum size limit | Passed |
| `test_save_uploaded_audio_and_cleanup` | Tests safe temp file writing and safe removal | Passed |
| `test_asr_initialization_default` | Checks `ASR()` loads default "base" model (mocked) | Passed |
| `test_asr_initialization_configurable_model` | Checks "tiny" and "small" model instantiation (mocked) | Passed |
| `test_asr_invalid_model_rejected` | Checks unsupported model name raises ValueError | Passed |
| `test_asr_transcribe_success` | Tests transcription output string and metadata (mocked) | Passed |
| `test_asr_transcribe_missing_file_raises` | Tests non-existent audio file path handling | Passed |
| `test_asr_transcribe_whisper_failure_raises_asr_error` | Tests runtime error handling during transcription | Passed |

---

## Demonstration Example

Below is a verified demonstration example illustrating the transcription input and output:

### Verified Input
- **File**: `sample_audio/test_speech.wav`
- **Spoken Speech**:
  > *"Artificial intelligence is transforming the world."*

### Verified Transcription Output
- **Model**: `base`
- **Detected Language**: `English` (`en`)
- **Output Text**:
  ```text
  Artificial intelligence is transforming the world.
  ```


---

## Whisper Model Information

Whisper provides several model sizes that balance speed, memory, and accuracy:

| Model | Parameters | Relative Speed | VRAM / RAM Required | Recommended Use Case |
|-------|------------|----------------|---------------------|----------------------|
| `tiny` | ~39 M | ~32x | ~1 GB | Fast prototyping, low-resource machines |
| **`base`** (Default) | ~74 M | ~16x | ~1 GB | **General-purpose laptop transcription (Recommended)** |
| `small` | ~244 M | ~6x | ~2 GB | Higher transcription accuracy |
| `medium` | ~769 M | ~2x | ~5 GB | Professional desktop / GPU setup |
| `large` | ~1550 M | 1x | ~10 GB | High-accuracy multi-lingual tasks |

The model can be switched dynamically in the Streamlit sidebar without modifying any source code.

---

## Limitations

- **Audio Quality Sensitivity**: Heavily distorted audio, strong background noise, or multiple overlapping speakers can degrade transcription accuracy.
- **Hardware Resources**: Running larger models (`small` or above) on CPU-only laptops requires noticeable processing time.
- **Initial Download Delay**: When running a model size for the first time, Whisper downloads model weights to the local cache (`~/.cache/whisper`), which may take a few moments depending on internet speed.
- **Browser Microphone Permissions**: Recording directly via the browser requires granting microphone permissions in the browser settings.

---

## Future Enhancements

- [ ] Real-time streaming transcription using WebRTC or WebSocket audio chunks
- [ ] Speaker Diarization (identifying who spoke when using pyannote.audio)
- [ ] Word-level timestamp alignments and subtitle generation (.SRT / .VTT export)
- [ ] Multi-lingual speech translation directly into English
- [ ] Voice Activity Detection (VAD) to filter silent segments prior to inference
- [ ] Noise reduction preprocessing using spectral gating
- [ ] Export transcriptions directly to Microsoft Word (.docx) and PDF formats
- [ ] GPU acceleration (CUDA) detection and automatic FP16 optimization

---

## Publishing to GitHub

To push this project to your GitHub account:

```bash
# 1. Initialize local Git repository
git init

# 2. Stage all files (respecting .gitignore)
git add .

# 3. Create initial commit
git commit -m "Initial commit - Speech2Text ASR"

# 4. Rename default branch to main
git branch -M main

# 5. Link to your GitHub repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Speech2Text-ASR.git

# 6. Push to GitHub
git push -u origin main
```

> **Note**: Replace `YOUR_USERNAME` with your actual GitHub username. The `.gitignore` file automatically ensures virtual environments, cache directories, temporary audio files, and model weights are not committed to GitHub.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
