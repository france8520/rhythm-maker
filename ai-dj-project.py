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
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
.stApp > header {
    background-color: transparent;
}
.stApp {
    color: white !important;
}
h1 {
    color: white;
    text-shadow: 0 0 10px #8A2BE2, 0 0 20px #8A2BE2, 0 0 30px #8A2BE2;
    font-size: 3em;
    margin-bottom: 30px;
}
.stButton > button {
    color: #4B0082;
    background-color: white;
    font-weight: bold;
}
.bubble-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}
.bubble {
    position: absolute;
    bottom: -100px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    opacity: 0.5;
    animation: rise 10s infinite ease-in;
}
@keyframes rise {
    0% {
        bottom: -100px;
        transform: translateX(0);
    }
    50% {
        transform: translate(100px, -500px);
    }
    100% {
        bottom: 1080px;
        transform: translateX(-200px);
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="bubble-container"></div>', unsafe_allow_html=True)


# Add JavaScript for bubble animation
st.markdown("""
<script>
function createBubbles() {
    const bubbleContainer = document.querySelector('.bubble-container');
    const bubbleCount = 50;

    for (let i = 0; i < bubbleCount; i++) {
        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        bubble.style.left = `${Math.random() * 100}%`;
        bubble.style.width = `${Math.random() * 30 + 10}px`;
        bubble.style.height = bubble.style.width;
        bubble.style.animationDuration = `${Math.random() * 15 + 5}s`;
        bubble.style.animationDelay = `${Math.random() * 5}s`;
        bubbleContainer.appendChild(bubble);
    }
}
document.addEventListener('DOMContentLoaded', createBubbles);
</script>
""", unsafe_allow_html=True)


os.environ['HF_TOKEN'] = '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'

# Rest of your Streamlit app code...
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

st.markdown("""
<style>
.button-container {
    display: flex;
    justify-content: center;
    gap: 5px;  /* Adjust this value to control the gap */
}
.stButton {
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# Wrap your buttons in a container div
st.markdown('<div class="button-container">', unsafe_allow_html=True)
for style in ["jazz", "rock", "electronic", "classical"]:
    if st.button(style.capitalize()):
        selected_style = style
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

if st.button("Make New Song"):
    st.rerun()