import streamlit as st
import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import time
import io
import wave
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip
import logging

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
.download-button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

os.environ['HF_TOKEN'] = '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'

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'], attn_implementation="eager").to(device)
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'])
        logging.info("Model and processor loaded successfully")
        return model, processor, device
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model, processor, device = load_model()

def generate_song(style, duration=15):
    try:
        prompt = f"Create an engaging {style} song with a catchy melody and rhythm"
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        sampling_rate = 32000
        total_samples = duration * sampling_rate
        max_new_tokens = min(int(total_samples / 512), 512)
        
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
    except Exception as e:
        logging.error(f"Error generating song: {str(e)}")
        st.error(f"Error generating song: {str(e)}")
        return None, None

def create_video(audio_data, sampling_rate, style):
    plt.figure(figsize=(10, 5))
    plt.plot(audio_data)
    plt.title(f"{style.capitalize()} Music Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('temp_waveform.png')
    plt.close()

    audio_clip = AudioFileClip.AudioArrayClip(audio_data, fps=sampling_rate)
    image_clip = ImageClip('temp_waveform.png').set_duration(audio_clip.duration)
    video = CompositeVideoClip([image_clip, audio_clip])
    video.write_videofile(f"{style}_music_video.mp4", fps=24)
    return f"{style}_music_video.mp4"

st.title("Rhythm Maker")
st.markdown('<p class="centered-text">Welcome to the AI DJ Project! Generate your own music with AI.</p>', unsafe_allow_html=True)

selected_style = st.selectbox("Choose a music style", ["Jazz", "Rock", "Electronic", "Classical"])

if st.button("Generate Music Video"):
    with st.spinner("Generating your music video..."):
        audio_data, sampling_rate = generate_song(selected_style.lower())
        if audio_data is not None and sampling_rate is not None:
            video_file = create_video(audio_data, sampling_rate, selected_style.lower())
            st.video(video_file)
            st.download_button("Download Video", video_file, file_name=f"{selected_style}_music_video.mp4")
        else:
            st.error("Failed to generate audio. Please try again.")

st.info("To generate a new video, please refresh the page.")
