import streamlit as st
import torch
import tempfile
from transformers import pipeline

st.set_page_config(page_title="Telugu Speech to Text", layout="centered")

st.title("🎙 Telugu Speech to Text (Whisper Telugu Large v2)")
st.write("Record your voice OR upload audio file")

# ----------------------------
# Load ASR Pipeline (Cached)
# ----------------------------
@st.cache_resource
def load_pipeline():
    device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        task="automatic-speech-recognition",
        model="vasista22/whisper-telugu-large-v2",
        chunk_length_s=30,
        device=device
    )

    # Force Telugu transcription
    asr.model.config.forced_decoder_ids = \
        asr.tokenizer.get_decoder_prompt_ids(
            language="te",
            task="transcribe"
        )

    return asr

asr = load_pipeline()

# ----------------------------
# Input Options
# ----------------------------
option = st.radio("Choose Input Method:", ["Upload Audio", "Record Audio"])

audio_file = None

if option == "Upload Audio":
    audio_file = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a"]
    )

elif option == "Record Audio":
    audio_file = st.audio_input("Record your voice")

# ----------------------------
# Transcription
# ----------------------------
if audio_file is not None:
    st.audio(audio_file)

    if st.button("🔍 Transcribe"):
        with st.spinner("Transcribing..."):

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            result = asr(tmp_path)
            transcription = result["text"]

        st.success("Transcription Complete!")
        st.text_area("📝 Telugu Text Output", transcription, height=200)
