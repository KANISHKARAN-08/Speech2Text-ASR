"""Streamlit Web Application for Speech2Text-ASR.

Provides an intuitive, professional web interface for transcribing uploaded
or recorded audio using OpenAI Whisper.
"""

import os
from datetime import datetime

import streamlit as st

from asr import ASR, ASRError
import config
from utils.audio_utils import (
    cleanup_temp_file,
    ensure_ffmpeg_on_path,
    save_uploaded_audio,
    validate_audio_file,
)

# Configure Streamlit page
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.15rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_asr_instance(model_name: str) -> ASR:
    """Load and cache the Whisper ASR model instance.

    Args:
        model_name: The name of the Whisper model ('tiny', 'base', 'small').

    Returns:
        ASR: Initialized ASR engine instance.
    """
    return ASR(model_name=model_name)


def main():
    """Main execution entry point for Streamlit application."""

    # Initialize session state for persistent results
    if "transcription" not in st.session_state:
        st.session_state["transcription"] = ""
    if "detected_language" not in st.session_state:
        st.session_state["detected_language"] = ""
    if "transcribed_file" not in st.session_state:
        st.session_state["transcribed_file"] = ""

    # Sidebar: Model Configuration & System Health
    st.sidebar.title("⚙️ Configuration")

    selected_model = st.sidebar.selectbox(
        "Select Whisper Model",
        options=config.AVAILABLE_MODELS,
        index=config.AVAILABLE_MODELS.index(config.DEFAULT_MODEL),
        help="Tiny is fastest with lowest memory; Base provides balanced speed and accuracy; Small offers higher precision.",
    )

    # Model specs display
    model_descriptions = {
        "tiny": "⚡ Fast inference (~39M params, low RAM)",
        "base": "⚖️ Balanced speed & accuracy (~74M params, default)",
        "small": "🎯 Higher accuracy (~244M params, higher RAM)",
    }
    st.sidebar.caption(model_descriptions.get(selected_model, ""))

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    ffmpeg_ready = ensure_ffmpeg_on_path()
    if ffmpeg_ready:
        st.sidebar.success("✅ FFmpeg is available")
    else:
        st.sidebar.error("❌ FFmpeg not detected in PATH")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "**Speech2Text-ASR**\n\n"
        "Automatic Speech Recognition tool powered by OpenAI Whisper and PyTorch.\n\n"
        "Built for offline/local audio transcription with zero external API dependencies."
    )

    # Main Application Header
    st.markdown(f"<div class='main-title'>🎙️ {config.APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-title'>{config.APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.write(config.APP_DESCRIPTION)

    if not ffmpeg_ready:
        st.warning(
            "⚠️ **Warning**: FFmpeg was not detected on system PATH. "
            "Please ensure FFmpeg is installed and added to PATH for proper audio processing."
        )

    st.markdown("---")

    # Mode Selection: Upload Audio or Record Audio
    mode = st.radio(
        "Select Input Mode:",
        options=["📁 Upload Audio", "🎙️ Record Audio"],
        horizontal=True,
    )

    audio_source = None
    file_display_name = ""

    if mode == "📁 Upload Audio":
        st.subheader("Upload Audio File")
        allowed_str = ", ".join(f.upper() for f in config.SUPPORTED_FORMATS)
        st.caption(f"Supported formats: **{allowed_str}** | Max size: **{config.MAX_FILE_SIZE_MB} MB**")

        audio_source = st.file_uploader(
            "Choose an audio file",
            type=config.SUPPORTED_FORMATS,
            help=f"Select an audio file formatted as {allowed_str}",
        )
        if audio_source is not None:
            file_display_name = audio_source.name

    else:
        st.subheader("Record Audio from Microphone")
        st.caption("Speak into your microphone to record speech for transcription.")

        if hasattr(st, "audio_input"):
            audio_source = st.audio_input("Record speech")
            if audio_source is not None:
                file_display_name = getattr(audio_source, "name", "microphone_recording.wav")
        else:
            st.warning(
                "Microphone recording requires Streamlit version 1.37 or newer (`st.audio_input`). "
                "Please use the **Upload Audio** tab to transcribe recorded files."
            )

    # Process and Transcribe Section
    if audio_source is not None:
        st.markdown("### Audio Preview")
        st.audio(audio_source)

        col_transcribe, col_reset = st.columns([3, 1])

        with col_transcribe:
            transcribe_button = st.button(
                "🎤 Transcribe Audio",
                type="primary",
                use_container_width=True,
            )

        with col_reset:
            reset_button = st.button("🔄 Reset", use_container_width=True)
            if reset_button:
                st.session_state["transcription"] = ""
                st.session_state["detected_language"] = ""
                st.session_state["transcribed_file"] = ""
                st.rerun()

        if transcribe_button:
            # Step 1: Validate file
            is_valid, error_msg = validate_audio_file(audio_source)
            if not is_valid:
                st.error(f"Validation Error: {error_msg}")
                return

            temp_path = None
            try:
                # Step 2: Save audio temporarily
                temp_path = save_uploaded_audio(audio_source)

                # Step 3: Load model and transcribe
                with st.spinner(f"Loading '{selected_model}' model and transcribing audio..."):
                    asr_engine = get_asr_instance(selected_model)
                    result = asr_engine.transcribe(temp_path, return_details=True)

                transcription_text = result.get("text", "").strip()
                detected_lang = result.get("language", "unknown")

                if not transcription_text:
                    st.warning("No intelligible speech was detected in the audio file.")
                    st.session_state["transcription"] = ""
                    st.session_state["detected_language"] = ""
                else:
                    st.session_state["transcription"] = transcription_text
                    st.session_state["detected_language"] = detected_lang
                    st.session_state["transcribed_file"] = file_display_name
                    st.success("Transcription completed successfully!")

            except ASRError as err:
                st.error(f"ASR Processing Error: {str(err)}")
            except Exception as err:
                st.error(f"Unexpected Error during transcription: {str(err)}")
            finally:
                if temp_path:
                    cleanup_temp_file(temp_path)

    # Display Transcription Results
    if st.session_state.get("transcription"):
        st.markdown("---")
        st.markdown("### 📝 Transcription")

        lang = st.session_state.get("detected_language")
        source_name = st.session_state.get("transcribed_file")
        if lang:
            st.caption(f"**Source:** `{source_name}` | **Detected Language:** `{lang.upper()}`")

        # Output text area
        st.text_area(
            label="Generated Transcription",
            value=st.session_state["transcription"],
            height=180,
            help="Transcribed text output from Whisper.",
        )

        # Download Transcription Option
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"transcription_{timestamp}.txt"

        download_content = (
            f"Speech2Text-ASR Transcription\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Source File: {source_name}\n"
            f"Detected Language: {lang}\n"
            f"Model: {selected_model}\n"
            f"----------------------------------------\n\n"
            f"{st.session_state['transcription']}\n"
        )

        st.download_button(
            label="📥 Download Transcription (.txt)",
            data=download_content,
            file_name=download_filename,
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
