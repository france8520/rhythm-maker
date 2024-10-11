import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import io
from scipy.io import wavfile
import logging
import base64
import asyncio
import uuid
from queue import Queue
import threading

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

# Global variables
model, processor, device = None, None, None
request_queue = Queue()
results = {}

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model.to(device)
    logging.info("Model and processor loaded successfully")
    return model, processor, device

async def generate_song(style, duration, session_id):
    prompt = f"Create an engaging {style} song with a catchy melody and rhythm"
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    sampling_rate = 32000
    total_samples = duration * sampling_rate
    max_new_tokens = min(int(total_samples / 320), 768)
    
    logging.info(f"Generating song with style: {style}, duration: {duration}s for session {session_id}")
    audio_values = await asyncio.to_thread(
        model.generate,
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        guidance_scale=3.0,
        temperature=1.0
    )
    
    audio_data = audio_values[0].cpu().numpy()
    audio_data = (audio_data * 32767).astype(np.int16)
    
    logging.info(f"Song generated successfully for session {session_id}")
    return audio_data, sampling_rate

def process_queue():
    while True:
        session_id, style, duration = request_queue.get()
        audio_data, sampling_rate = asyncio.run(generate_song(style, duration, session_id))
        results[session_id] = (audio_data, sampling_rate)
        request_queue.task_done()

def get_audio_download_link(audio_data, sampling_rate, filename):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, sampling_rate, audio_data)
    virtualfile.seek(0)
    b64 = base64.b64encode(virtualfile.getvalue()).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    global model, processor, device

    if model is None:
        model, processor, device = load_model()

    st.title("Rhythm Maker")
    st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'request_submitted' not in st.session_state:
        st.session_state.request_submitted = False

    selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical"])
    duration = st.slider("Select duration (seconds)", 5, 30, 15)

    if st.button("Generate Music"):
        if not st.session_state.request_submitted:
            request_queue.put((st.session_state.session_id, selected_style.lower(), duration))
            st.session_state.request_submitted = True
            st.info("Your request has been submitted. Please wait...")

    if st.session_state.request_submitted:
        if st.session_state.session_id in results:
            audio_data, sampling_rate = results[st.session_state.session_id]
            
            audio_buffer = io.BytesIO()
            wavfile.write(audio_buffer, sampling_rate, audio_data)
            audio_buffer.seek(0)
            
            st.audio(audio_buffer, format='audio/wav')
            st.markdown(get_audio_download_link(audio_data, sampling_rate, f"{selected_style.lower()}_music.wav"), unsafe_allow_html=True)
            
            # Clear the result and reset the session
            del results[st.session_state.session_id]
            st.session_state.request_submitted = False
            st.session_state.session_id = str(uuid.uuid4())
        else:
            st.info("Your music is being generated. Please wait...")

if __name__ == "__main__":
    # Start the queue processing thread
    threading.Thread(target=process_queue, daemon=True).start()
    main()