"""Automatic Speech Recognition (ASR) engine using OpenAI Whisper.

Provides an independent and reusable interface for speech-to-text processing.
"""

import os
from typing import Dict, Optional, Union

import whisper

import config
from utils.audio_utils import ensure_ffmpeg_on_path


class ASRError(Exception):
    """Custom exception raised for ASR pipeline errors."""


class ASR:
    """Automatic Speech Recognition handler wrapping OpenAI Whisper models."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        """Initialize ASR engine and load the specified Whisper model.

        Args:
            model_name: Name of the Whisper model ('tiny', 'base', 'small').

        Raises:
            ValueError: If the requested model is not supported.
            ASRError: If model loading fails.
        """
        if model_name not in config.AVAILABLE_MODELS:
            valid_models = ", ".join(config.AVAILABLE_MODELS)
            raise ValueError(
                f"Unsupported model '{model_name}'. Choose from: {valid_models}"
            )

        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str) -> whisper.Whisper:
        """Load Whisper model from cache or download.

        Args:
            model_name: Name of the Whisper model.

        Returns:
            whisper.Whisper: Loaded Whisper model instance.

        Raises:
            ASRError: If loading fails due to network, memory, or torch issues.
        """
        try:
            return whisper.load_model(model_name)
        except Exception as exc:
            raise ASRError(
                f"Failed to load Whisper model '{model_name}': {str(exc)}"
            ) from exc

    def transcribe(
        self, audio_file: str, return_details: bool = False
    ) -> Union[str, Dict]:
        """Transcribe speech from an audio file into text.

        Args:
            audio_file: File path to the audio file.
            return_details: If True, returns full transcription dictionary with
                metadata (language, segments). If False, returns transcription text string.

        Returns:
            Union[str, Dict]: Transcription text string or detailed dictionary.

        Raises:
            FileNotFoundError: If audio file does not exist on disk.
            ASRError: If FFmpeg is missing or Whisper transcription fails.
        """
        if not audio_file or not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if not ensure_ffmpeg_on_path():
            raise ASRError(
                "FFmpeg is not installed or not found in system PATH. "
                "FFmpeg is required by Whisper for audio decoding."
            )

        try:
            # Perform Whisper inference
            result = self.model.transcribe(audio_file)
            transcription_text = result.get("text", "").strip()

            if return_details:
                return {
                    "text": transcription_text,
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", []),
                }

            return transcription_text

        except Exception as exc:
            raise ASRError(f"Transcription failed: {str(exc)}") from exc
