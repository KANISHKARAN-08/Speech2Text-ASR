"""Application Configuration for Speech2Text-ASR.

Centralizes settings, model definitions, supported formats, and UI constants.
"""

# Application Branding & Metadata
APP_TITLE = "Speech2Text ASR"
APP_SUBTITLE = "Automatic Speech Recognition Tool"
APP_DESCRIPTION = (
    "Convert speech into text using OpenAI Whisper. "
    "Upload an audio file or record from your microphone to generate accurate transcriptions."
)

# ASR Model Settings
DEFAULT_MODEL = "base"
AVAILABLE_MODELS = ["tiny", "base", "small"]

# Audio Settings
SUPPORTED_FORMATS = ["wav", "mp3", "m4a", "ogg", "flac"]
MAX_FILE_SIZE_MB = 50

# Storage Settings
TEMP_AUDIO_DIR = "temp_audio"
