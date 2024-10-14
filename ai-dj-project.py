import streamlit as st
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import numpy as np
import io
from scipy.io import wavfile
import scipy.signal as signal
import logging
import base64
import time
import uuid
import os


# Set up logging
logging.basicConfig(level=logging.INFO)

# Check and set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Streamlit configuration
st.set_page_config(page_title="Universal Rhythm Maker", layout="wide")
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

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
    model.to(device)
    return model, processor, device

def generate_song(model, processor, device, style, duration=15):
    prompt = f"Create a short {style} melody"
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_length = int(duration * sampling_rate)
    
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=int(duration * 25),
            do_sample=True,
            guidance_scale=2.0,
            temperature=0.8
        )
    
    audio_data = audio_values[0].cpu().float().numpy()
    audio_data = np.pad(audio_data, (0, max(0, audio_length - len(audio_data))))[:audio_length]
    audio_data = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)
    
    return audio_data, sampling_rate

def main():
    st.title("Rhythm Maker")
    st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

    selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical", "Pop"])
    duration = st.selectbox("Select duration (seconds)", [5, 10, 15, 20, 30])

    if st.button("Generate Music"):
        model, processor, device = load_model()
        
        with st.spinner("Generating your music..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.03)
                progress_bar.progress(i + 1)
            
            audio_data, sampling_rate = generate_song(model, processor, device, selected_style.lower(), duration)
            
            if audio_data is not None:
                audio_buffer = io.BytesIO()
                wavfile.write(audio_buffer, sampling_rate, audio_data)
                audio_buffer.seek(0)
                
                st.audio(audio_buffer, format='audio/wav')
                st.success("Music generated successfully!")
            else:
                st.error("Failed to generate music. Please try again.")

if __name__ == "__main__":
    main()
