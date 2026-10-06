"""Audio utilities for validating, saving, and cleaning up audio files.

Provides safe temporary file management and environment checks for FFmpeg.
"""

import os
import shutil
import uuid
from typing import Tuple, Union

import config


def ensure_ffmpeg_on_path() -> bool:
    """Ensure FFmpeg executable is discoverable in PATH.

    If not found in standard system PATH, searches common Windows locations
    and optional imageio_ffmpeg fallback.

    Returns:
        bool: True if ffmpeg executable is found and ready, False otherwise.
    """
    if shutil.which("ffmpeg"):
        return True

    # Check imageio_ffmpeg fallback if available
    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_exe):
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg"):
                return True
    except ImportError:
        pass

    # Check common Windows winget installation path for Gyan.FFmpeg
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_ffmpeg_dir = os.path.join(
            local_app_data, "Microsoft", "WinGet", "Packages"
        )
        if os.path.isdir(winget_ffmpeg_dir):
            for root, _, files in os.walk(winget_ffmpeg_dir):
                if "ffmpeg.exe" in files:
                    os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                    return True

    return bool(shutil.which("ffmpeg"))


def validate_audio_file(
    file_obj_or_name: Union[str, any],
    supported_formats: list = None,
    max_size_mb: int = None,
) -> Tuple[bool, str]:
    """Validate audio file extension and size constraints.

    Args:
        file_obj_or_name: File path, filename, or Streamlit UploadedFile object.
        supported_formats: List of allowed lowercase file extensions.
        max_size_mb: Maximum allowed file size in Megabytes.

    Returns:
        Tuple[bool, str]: (is_valid, error_message). If valid, error_message is empty.
    """
    if supported_formats is None:
        supported_formats = config.SUPPORTED_FORMATS
    if max_size_mb is None:
        max_size_mb = config.MAX_FILE_SIZE_MB

    if file_obj_or_name is None:
        return False, "No audio file provided."

    # Extract filename
    if hasattr(file_obj_or_name, "name"):
        filename = file_obj_or_name.name
    elif isinstance(file_obj_or_name, str):
        filename = file_obj_or_name
    else:
        return False, "Invalid audio input format."

    if not filename.strip():
        return False, "File name cannot be empty."

    # Validate extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not ext or ext not in supported_formats:
        allowed = ", ".join(fmt.upper() for fmt in supported_formats)
        return False, f"Unsupported file format '.{ext}'. Supported formats: {allowed}"

    # Validate file size if available
    size_bytes = None
    if hasattr(file_obj_or_name, "size"):
        size_bytes = file_obj_or_name.size
    elif isinstance(file_obj_or_name, str) and os.path.exists(file_obj_or_name):
        size_bytes = os.path.getsize(file_obj_or_name)

    if size_bytes is not None:
        if size_bytes == 0:
            return False, "Uploaded file is empty (0 bytes)."

        max_bytes = max_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            actual_mb = size_bytes / (1024 * 1024)
            return (
                False,
                f"File size ({actual_mb:.1f} MB) exceeds maximum limit of {max_size_mb} MB.",
            )

    return True, ""


def save_uploaded_audio(uploaded_file, target_dir: str = None) -> str:
    """Save an uploaded audio file or byte buffer safely to a unique temporary file.

    Args:
        uploaded_file: Streamlit UploadedFile or object with audio bytes.
        target_dir: Directory where temporary files are stored.

    Returns:
        str: Absolute path to the saved temporary audio file.

    Raises:
        ValueError: If uploaded_file is empty or invalid.
        IOError: If writing to disk fails.
    """
    if uploaded_file is None:
        raise ValueError("Cannot save None audio object.")

    if target_dir is None:
        target_dir = config.TEMP_AUDIO_DIR

    os.makedirs(target_dir, exist_ok=True)

    # Determine file extension
    original_name = getattr(uploaded_file, "name", "recorded_audio.wav")
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else "wav"
    unique_filename = f"audio_{uuid.uuid4().hex[:10]}.{ext}"
    dest_path = os.path.abspath(os.path.join(target_dir, unique_filename))

    # Read binary content
    if hasattr(uploaded_file, "getbuffer"):
        data = uploaded_file.getbuffer()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    elif isinstance(uploaded_file, (bytes, bytearray)):
        data = uploaded_file
    else:
        raise ValueError("Unsupported audio data structure.")

    with open(dest_path, "wb") as f:
        f.write(data)

    return dest_path


def cleanup_temp_file(file_path: str) -> bool:
    """Safely remove a temporary file.

    Args:
        file_path: Path to the temporary file to delete.

    Returns:
        bool: True if deleted or did not exist, False if deletion failed.
    """
    if not file_path:
        return False

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except OSError:
        return False


def cleanup_temp_dir(target_dir: str = None) -> None:
    """Clean up all transient audio files inside target directory.

    Args:
        target_dir: Directory containing temporary audio files.
    """
    if target_dir is None:
        target_dir = config.TEMP_AUDIO_DIR

    if not os.path.isdir(target_dir):
        return

    for entry in os.listdir(target_dir):
        full_path = os.path.join(target_dir, entry)
        if os.path.isfile(full_path):
            cleanup_temp_file(full_path)
