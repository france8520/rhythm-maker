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
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")  # Reverted to small model for faster loading
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model.to(device)
        logging.info("Model and processor loaded successfully")
        return model, processor, device
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        st.error("Failed to load the AI model. Please try again later.")
        return None, None, None

def generate_song(model, processor, device, style, duration=20):
    try:
        prompt = f"Create a {style} melody"
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        sampling_rate = 24000  # Adjusted sampling rate
        total_samples = duration * sampling_rate
        max_new_tokens = min(int(total_samples / 320), 256)
        
        logging.info(f"Generating song with style: {style}, duration: {duration}s")
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=3.0,
                temperature=1.0
            )
        
        audio_data = audio_values[0].cpu().numpy()
        
        # Basic post-processing
        audio_data = signal.sosfilt(signal.butter(10, 100, 'hp', fs=sampling_rate, output='sos'), audio_data)
        audio_data = signal.sosfilt(signal.butter(10, 10000, 'lp', fs=sampling_rate, output='sos'), audio_data)
        
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = (audio_data * 32767).astype(np.int16)
        
        logging.info("Song generated successfully")
        return audio_data, sampling_rate
    except Exception as e:
        logging.error(f"Error generating song: {str(e)}")
        return None, None

def get_audio_download_link(audio_data, sampling_rate, filename):
    try:
        virtualfile = io.BytesIO()
        wavfile.write(virtualfile, sampling_rate, audio_data.T)
        virtualfile.seek(0)
        b64 = base64.b64encode(virtualfile.getvalue()).decode()
        return f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'
    except Exception as e:
        logging.error(f"Error creating download link: {str(e)}")
        return None

@st.cache_data(ttl=3600, max_entries=50)
def cached_generate_song(style, duration, user_id):
    model, processor, device = load_model()
    if model is None:
        return None, None
    return generate_song(model, processor, device, style, duration)

def main():
    st.title("Universal Rhythm Maker")
    st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

    selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical", "Pop"])
    duration = st.slider("Select duration (seconds)", 5, 20, 10)

    if st.button("Generate Music"):
        try:
            with st.spinner("Generating your music... This may take a few minutes."):
                start_time = time.time()
                
                audio_data, sampling_rate = cached_generate_song(selected_style.lower(), duration, st.session_state.user_id)
                
                if audio_data is None:
                    st.error("Failed to generate music. Please try again.")
                    return

                end_time = time.time()
                elapsed_time = end_time - start_time
                
                audio_buffer = io.BytesIO()
                wavfile.write(audio_buffer, sampling_rate, audio_data.T)
                audio_buffer.seek(0)
                
                st.audio(audio_buffer, format='audio/wav')
                
                download_link = get_audio_download_link(audio_data, sampling_rate, f"{selected_style.lower()}_music_{st.session_state.user_id}.wav")
                if download_link:
                    st.markdown(download_link, unsafe_allow_html=True)
                
                st.success(f"Music generated in {elapsed_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()