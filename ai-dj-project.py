import streamlit as st
import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np

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
.button-container {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 20px;
}
.stButton > button {
    color: #4B0082;
    background-color: white;
    font-weight: bold;
    width: 120px;  /* Set a fixed width for all buttons */
}
</style>
""", unsafe_allow_html=True)

os.environ['HF_TOKEN'] = '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'

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'], attn_implementation="eager").to(device)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'])
    return model, processor, device

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
    
    max_new_tokens = min(int(total_samples / 256), 1024)  # Limit to 1024 tokens
    
    audio_values = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        guidance_scale=3.5,
        temperature=0.8
    )
    
    audio_data = audio_values[0].cpu().numpy()
    
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data, sampling_rate

st.title("Rhythm Maker")
st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

# Wrap your buttons in a container div
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if col1.button("Jazz"):
    selected_style = "jazz"
if col2.button("Rock"):
    selected_style = "rock"
if col3.button("Electronic"):
    selected_style = "electronic"
if col4.button("Classical"):
    selected_style = "classical"
st.markdown('</div>', unsafe_allow_html=True)

if 'selected_style' in locals():
    with st.spinner("Generating your song..."):
        audio_data, sampling_rate = generate_song(selected_style)
    
    st.audio(audio_data, format='audio/wav', sample_rate=sampling_rate)
    
    st.download_button(
        label="Download Song",
        data=audio_data.tobytes(),
        file_name=f"{selected_style}_song.wav",
        mime="audio/wav"
    )

# Center the "Make New Song" button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("Make New Song"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)