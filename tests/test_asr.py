"""Unit tests for Speech2Text-ASR using pytest.

Tests configuration, audio validation, temporary file management,
and Whisper ASR initialization and transcription logic (with mocking).
"""

import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from asr import ASR, ASRError
import config
from utils.audio_utils import (
    cleanup_temp_file,
    save_uploaded_audio,
    validate_audio_file,
)


class DummyUploadedFile:
    """Mock Streamlit UploadedFile object for unit testing."""

    def __init__(self, name: str, data: bytes, size: int = None):
        self.name = name
        self.data = data
        self.size = size if size is not None else len(data)

    def getbuffer(self):
        return self.data

    def read(self):
        return self.data


# 1. Configuration Tests
def test_config_values():
    """Verify that configuration constants match requirements."""
    assert config.APP_TITLE == "Speech2Text ASR"
    assert config.DEFAULT_MODEL == "base"
    assert "base" in config.AVAILABLE_MODELS
    assert "tiny" in config.AVAILABLE_MODELS
    assert "small" in config.AVAILABLE_MODELS
    assert set(config.SUPPORTED_FORMATS) == {"wav", "mp3", "m4a", "ogg", "flac"}
    assert config.MAX_FILE_SIZE_MB > 0


# 2. Audio File Validation Tests
@pytest.mark.parametrize(
    "filename",
    [
        "speech.wav",
        "voice.mp3",
        "audio.m4a",
        "recording.ogg",
        "sample.flac",
        "UPPERCASE.WAV",
        "MixedCase.Mp3",
    ],
)
def test_validate_audio_file_valid_extensions(filename):
    """Ensure all required audio extensions are accepted."""
    is_valid, msg = validate_audio_file(filename)
    assert is_valid is True
    assert msg == ""


@pytest.mark.parametrize(
    "filename",
    [
        "document.txt",
        "video.mp4",
        "script.py",
        "archive.zip",
        "recording.aac",
        "no_extension",
    ],
)
def test_validate_audio_file_invalid_extensions(filename):
    """Ensure non-supported file extensions are rejected."""
    is_valid, msg = validate_audio_file(filename)
    assert is_valid is False
    assert "Unsupported file format" in msg


def test_validate_audio_file_empty_and_none():
    """Verify handling of None and empty inputs."""
    is_valid, msg = validate_audio_file(None)
    assert is_valid is False
    assert "No audio file provided" in msg

    is_valid, msg = validate_audio_file("")
    assert is_valid is False
    assert "File name cannot be empty" in msg


def test_validate_audio_file_zero_bytes():
    """Verify rejection of 0-byte audio files."""
    dummy = DummyUploadedFile(name="empty.wav", data=b"", size=0)
    is_valid, msg = validate_audio_file(dummy)
    assert is_valid is False
    assert "0 bytes" in msg


def test_validate_audio_file_size_exceeded():
    """Verify rejection of audio files exceeding max size limit."""
    oversized_bytes = (config.MAX_FILE_SIZE_MB + 5) * 1024 * 1024
    dummy = DummyUploadedFile(name="huge.wav", data=b"x", size=oversized_bytes)
    is_valid, msg = validate_audio_file(dummy)
    assert is_valid is False
    assert "exceeds maximum limit" in msg


# 3. Audio File Storage and Cleanup Tests
def test_save_uploaded_audio_and_cleanup(tmp_path):
    """Test saving audio buffer to disk and deleting it safely."""
    test_data = b"RIFF....WAVEfmt ...."
    dummy = DummyUploadedFile(name="test_clip.wav", data=test_data)

    target_dir = str(tmp_path / "test_audio_dir")
    saved_path = save_uploaded_audio(dummy, target_dir=target_dir)

    assert os.path.exists(saved_path)
    assert saved_path.endswith(".wav")
    with open(saved_path, "rb") as f:
        assert f.read() == test_data

    # Test cleanup
    assert cleanup_temp_file(saved_path) is True
    assert not os.path.exists(saved_path)

    # Calling cleanup on non-existent file should safely return True without error
    assert cleanup_temp_file("non_existent_file.wav") is True


def test_save_uploaded_audio_none_raises():
    """Test that saving None audio object raises ValueError."""
    with pytest.raises(ValueError):
        save_uploaded_audio(None)


# 4. ASR Engine Initialization Tests (Mocked)
@patch("whisper.load_model")
def test_asr_initialization_default(mock_load_model):
    """Verify ASR initializes default 'base' model without downloading."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    asr_engine = ASR()

    assert asr_engine.model_name == "base"
    mock_load_model.assert_called_once_with("base")


@patch("whisper.load_model")
def test_asr_initialization_configurable_model(mock_load_model):
    """Verify ASR accepts configurable models: tiny, small."""
    mock_load_model.return_value = MagicMock()

    asr_tiny = ASR(model_name="tiny")
    assert asr_tiny.model_name == "tiny"
    mock_load_model.assert_called_with("tiny")

    asr_small = ASR(model_name="small")
    assert asr_small.model_name == "small"
    mock_load_model.assert_called_with("small")


def test_asr_invalid_model_rejected():
    """Verify that unsupported model names raise ValueError."""
    with pytest.raises(ValueError) as excinfo:
        ASSR = ASR(model_name="large-v99")
    assert "Unsupported model" in str(excinfo.value)


# 5. ASR Transcription Tests (Mocked)
@patch("whisper.load_model")
def test_asr_transcribe_success(mock_load_model, tmp_path):
    """Verify transcription process with mocked Whisper model."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": " Artificial intelligence is transforming the world. ",
        "language": "en",
        "segments": [{"id": 0, "text": "Artificial intelligence is transforming the world."}],
    }
    mock_load_model.return_value = mock_model

    fake_audio = tmp_path / "valid_speech.wav"
    fake_audio.write_bytes(b"dummy audio content")

    asr_engine = ASR(model_name="base")
    transcription = asr_engine.transcribe(str(fake_audio))

    assert transcription == "Artificial intelligence is transforming the world."

    # Test detailed output
    details = asr_engine.transcribe(str(fake_audio), return_details=True)
    assert isinstance(details, dict)
    assert details["text"] == "Artificial intelligence is transforming the world."
    assert details["language"] == "en"


@patch("whisper.load_model")
def test_asr_transcribe_missing_file_raises(mock_load_model):
    """Verify FileNotFoundError when input audio path does not exist."""
    mock_load_model.return_value = MagicMock()
    asr_engine = ASR()

    with pytest.raises(FileNotFoundError):
        asr_engine.transcribe("does_not_exist_on_disk.wav")


@patch("whisper.load_model")
def test_asr_transcribe_whisper_failure_raises_asr_error(mock_load_model, tmp_path):
    """Verify runtime whisper failure is wrapped in ASRError."""
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory or corrupted audio")
    mock_load_model.return_value = mock_model

    fake_audio = tmp_path / "corrupt_speech.wav"
    fake_audio.write_bytes(b"corrupt")

    asr_engine = ASR()

    with pytest.raises(ASRError) as excinfo:
        asr_engine.transcribe(str(fake_audio))
    assert "Transcription failed" in str(excinfo.value)
