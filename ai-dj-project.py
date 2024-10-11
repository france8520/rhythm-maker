import streamlit as st
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import numpy as np
import io
from scipy.io import wavfile
import logging
import base64
import time
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)

# Streamlit configuration
st.set_page_config(page_title="Fast Rhythm Maker", layout="wide")
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
    device = "cpu"  # Force CPU for consistent performance across all devices
    logging.info(f"Using device: {device}")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
    model.to(device)
    logging.info("Model and processor loaded successfully")
    return model, processor, device

def generate_song(model, processor, device, style, duration=15):
    prompt = f"Short {style} melody"
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    sampling_rate = 32000
    max_new_tokens = min(int(duration * sampling_rate / 256), 512)  # Adjust tokens based on duration
    
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
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Trim or pad audio to exact duration
    target_samples = duration * sampling_rate
    if audio_data.shape[0] > target_samples:
        audio_data = audio_data[:target_samples]
    elif audio_data.shape[0] < target_samples:
        audio_data = np.pad(audio_data, (0, target_samples - audio_data.shape[0]), mode='constant')
    
    logging.info("Song generated successfully")
    return audio_data, sampling_rate

def get_audio_download_link(audio_data, sampling_rate, filename):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, sampling_rate, audio_data.T)
    virtualfile.seek(0)
    b64 = base64.b64encode(virtualfile.getvalue()).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'

@st.cache_data(ttl=1800, max_entries=20)  # Cache for 30 minutes, limit to 20 entries
def cached_generate_song(style, duration, user_id):
    model, processor, device = load_model()
    return generate_song(model, processor, device, style, duration)

def main():
    st.title("Fast Rhythm Maker")
    st.markdown('<p class="centered-text">Generate quick musical snippets with AI.</p>', unsafe_allow_html=True)

    selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical"])
    duration = st.slider("Select duration (seconds)", 5, 15, 10)  # Adjusted max duration to 15 seconds

    if st.button("Generate Music"):
        try:
            with st.spinner("Generating your music snippet..."):
                start_time = time.time()
                
                # Use cached function to generate or retrieve cached audio
                audio_data, sampling_rate = cached_generate_song(selected_style.lower(), duration, st.session_state.user_id)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Create a BytesIO object to store the audio data
                audio_buffer = io.BytesIO()
                wavfile.write(audio_buffer, sampling_rate, audio_data.T)
                audio_buffer.seek(0)
                
                # Display audio player
                st.audio(audio_buffer, format='audio/wav')
                
                # Provide download link
                st.markdown(get_audio_download_link(audio_data, sampling_rate, f"{selected_style.lower()}_snippet_{st.session_state.user_id}.wav"), unsafe_allow_html=True)
                
                # Display generation time
                st.success(f"Music generated in {elapsed_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred while generating the music: {str(e)}")
            logging.error(f"Error generating music: {str(e)}")

if __name__ == "__main__":
    main()