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
        
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_length = int(duration * sampling_rate)
        
        logging.info(f"Generating song with style: {style}, duration: {duration}s")
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                guidance_scale=3.0,
                temperature=1.0
            )
        
        logging.info(f"Shape of generated audio: {audio_values.shape}")
        
        audio_data = audio_values[0].cpu().float().numpy()
        
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        elif audio_data.ndim > 2:
            audio_data = audio_data.mean(axis=0, keepdims=True)
        
        if audio_data.shape[1] < audio_length:
            padding = np.zeros((audio_data.shape[0], audio_length - audio_data.shape[1]))
            audio_data = np.concatenate([audio_data, padding], axis=1)
        elif audio_data.shape[1] > audio_length:
            audio_data = audio_data[:, :audio_length]
        
        for i in range(audio_data.shape[0]):
            audio_data[i] = signal.sosfilt(signal.butter(10, 100, 'hp', fs=sampling_rate, output='sos'), audio_data[i])
            audio_data[i] = signal.sosfilt(signal.butter(10, 10000, 'lp', fs=sampling_rate, output='sos'), audio_data[i])
        
        # Normalize audio to [-1, 1] range
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Convert to int16 range [-32768, 32767]
        audio_data = (audio_data * 32767).astype(np.int16)
        
        logging.info(f"Final audio shape: {audio_data.shape}")
        logging.info(f"Audio data range: {audio_data.min()} to {audio_data.max()}")
        logging.info("Song generated successfully")
        
        return audio_data, sampling_rate
    except Exception as e:
        logging.error(f"Error generating song: {str(e)}")
        return None, None
def generate_song_wrapper(style, duration, user_id):
    model, processor, device = load_model()
    if model is None:
        return None, None
    return generate_song(model, processor, device, style, duration)

@st.cache_data(ttl=3600, max_entries=50)
def cached_generate_song(style, duration, user_id):
    model, processor, device = load_model()
    if model is None:
        return None, None
    return generate_song(model, processor, device, style, duration)

def main():
    st.title("Rhythm Maker")
    st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

    selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical", "Pop"])
    
    # Replace slider with a select box for duration
    duration_options = [5, 10]
    duration = st.selectbox("Select duration (seconds)", duration_options)

    if st.button("Generate Music"):
        try:
            with st.spinner("Generating your music... This may take a few minutes."):
                start_time = time.time()
                
                # Use generate_song_wrapper instead of cached_generate_song
                audio_data, sampling_rate = generate_song_wrapper(selected_style.lower(), duration, st.session_state.user_id)
                
                if audio_data is None:
                    st.error("Failed to generate music. Please try again.")
                    return

                end_time = time.time()
                elapsed_time = end_time - start_time
                
                logging.info(f"Audio data shape: {audio_data.shape}")
                logging.info(f"Audio data type: {audio_data.dtype}")
                logging.info(f"Sampling rate: {sampling_rate}")
                
                audio_buffer = io.BytesIO()
                wavfile.write(audio_buffer, sampling_rate, audio_data.T)
                audio_buffer.seek(0)
                
                st.audio(audio_buffer, format='audio/wav')
                
                st.success(f"Music generated in {elapsed_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
