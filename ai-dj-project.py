import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import streamlit as st

os.environ['HF_TOKEN'] = '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'
# Increase startup timeout
st.set_page_config(page_title="Rhythm Maker", layout="wide")

# Load CSS from index.html
with open('templates/index.html', 'r') as file:
    html_content = file.read()
    css_content = html_content.split('<style>')[1].split('</style>')[0]

# Apply custom CSS including background
st.markdown(f"""
<style>
{css_content}
body {{
    background: linear-gradient(90deg, #8A2BE2 0%, #4B0082 30%, #000000 100%);
}}
</style>
""", unsafe_allow_html=True)

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
st.write("Welcome to the AI DJ Project! Generate your own music with AI.")

styles = ["jazz", "rock", "electronic", "classical"]
cols = st.columns(len(styles))
for i, style in enumerate(styles):
    if cols[i].button(style.capitalize()):
        selected_style = style

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

st.button("Make New Song", on_click=lambda: st.experimental_rerun())

if st.button("Make New Song"):
    st.rerun()
