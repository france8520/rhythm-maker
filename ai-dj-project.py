import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import io
import soundfile as sf
import logging
import base64
from scipy.io import wavfile

# Set up logging
logging.basicConfig(level=logging.INFO)

# Streamlit configuration
st.set_page_config(page_title="Rhythm Maker", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(90deg, #8A2BE2 0%, #4B0082 30%, #000000 100%);
    color: white !important;
}
h1 {
    color: white;
    text-shadow: 0 0 10px #8A2BE2, 0 0 20px #8A2BE2, 0 0 30px #8A2BE2;
    font-size: 3em;
    margin-bottom: 30px;
    text-align: center;
}
.centered-text {
    text-align: center;
}
.stButton > button {
    color: #4B0082;
    background-color: white;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    logging.info("Model and processor loaded successfully")
    return model, processor, device

model, processor, device = load_model()

def generate_song(style, duration=15):
    prompt = f"Create an engaging {style} song with a catchy melody and rhythm"
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    sampling_rate = 32000
    total_samples = duration * sampling_rate
    max_new_tokens = min(int(total_samples / 256), 1024)
    
    logging.info(f"Generating song with style: {style}, duration: {duration}s")
    audio_values = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        guidance_scale=3.0,
        temperature=1.0
    )
    
    audio_data = audio_values[0].cpu().numpy()
    audio_data = (audio_data * 32767).astype(np.int16)
    
    logging.info("Song generated successfully")
    return audio_data, sampling_rate

def get_audio_download_link(audio_data, sampling_rate, filename):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, sampling_rate, audio_data.T)
    virtualfile.seek(0)
    b64 = base64.b64encode(virtualfile.getvalue()).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    st.title("Rhythm Maker")
    st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

    selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical"])
    duration = st.slider("Select duration (seconds)", 5, 30, 15)

    if st.button("Generate Music"):
        with st.spinner("Generating your music..."):
            audio_data, sampling_rate = generate_song(selected_style.lower(), duration)
            
            # Create a BytesIO object to store the audio data
            audio_buffer = io.BytesIO()
            wavfile.write(audio_buffer, sampling_rate, audio_data.T)
            audio_buffer.seek(0)
            
            # Display audio player
            st.audio(audio_buffer, format='audio/wav')
            
            # Provide download link
            st.markdown(get_audio_download_link(audio_data, sampling_rate, f"{selected_style.lower()}_music.wav"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()