import streamlit as st
import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import time
# Set page config first
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
div[data-testid="stHorizontalBlock"] {
    gap: 0rem !important;
}
.button-container {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 20px;
}
.stButton > button {
    min-width: 100px;
}
</style>
""", unsafe_allow_html=True)

os.environ['HF_TOKEN'] = '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'

@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'], attn_implementation="eager").to(device)
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'])
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model, processor, device = load_model()

def generate_song(style, duration=60):
    prompt = f"Create an engaging {style} song with a catchy melody and rhythm"
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    sampling_rate = 32000
    total_samples = duration * sampling_rate
    max_new_tokens = min(int(total_samples / 256), 1024)
    
    progress_bar = st.progress(0)
    
    audio_values = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        guidance_scale=3.5,
        temperature=0.8
    )
    
    progress_bar.progress(1.0)  # Set progress to 100% after generation
    
    audio_data = audio_values[0].cpu().numpy()
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data, sampling_rate

st.title("Rhythm Maker")
st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical"])

if st.button("Generate Song"):
    with st.spinner("Preparing to generate your song..."):
        # Add a small delay to ensure the spinner is visible
        time.sleep(1)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for i in range(10):  # Simulate progress
            status_text.text(f"Generating song... Step {i+1}/10")
            progress_bar.progress((i + 1) * 10)
            time.sleep(0.5)  # Simulate work being done

        audio_data, sampling_rate = generate_song(selected_style.lower())
        
        status_text.text("Song generated successfully!")
        progress_bar.progress(100)

        st.audio(audio_data, format='audio/wav', sample_rate=sampling_rate)
        
        st.download_button(
            label="Download Song",
            data=audio_data.tobytes(),
            file_name=f"{selected_style.lower()}_song.wav",
            mime="audio/wav"
        )
    except Exception as e:
        st.error(f"An error occurred while generating the song: {str(e)}")

    # Clear the progress bar and status text after completion or error
    progress_bar.empty()
    status_text.empty()

if st.button("Make New Song"):
    st.rerun()