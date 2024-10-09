import streamlit as st
import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np

# Set page config first
st.set_page_config(page_title="Rhythm Maker", layout="wide")

# Now load CSS and apply it
with open('templates/index.html', 'r') as file:
    html_content = file.read()
    css_content = html_content.split('<style>')[1].split('</style>')[0]

# Update the CSS for a more readable and visually appealing style
st.markdown('''
<style>
.stApp {
    background: linear-gradient(90deg, #003366 0%, #000066 50%, #000000 100%);
}
.stApp > header {
    background-color: transparent;
}
.stApp {
    color: white !important;
    font-family: 'Arial', sans-serif;
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
    opacity: 0.3;
    animation: rise 10s infinite ease-in;
}
@keyframes rise {
    0% {
        bottom: -100px;
        transform: translateX(0);
    }
    50% {
        transform: translate(50px, -500px);
    }
    100% {
        bottom: 1080px;
        transform: translateX(-150px);
    }
}
</style>
<div class="bubble-container"></div>
<script>
function createBubbles() {
    const bubbleContainer = document.querySelector('.bubble-container');
    const bubbleCount = 30; // Reduced bubble count for less clutter
    for (let i = 0; i < bubbleCount; i++) {
        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        bubble.style.left = `${Math.random() * 100}%`;
        bubble.style.width = `${Math.random() * 25 + 10}px`;
        bubble.style.height = bubble.style.width;
        bubble.style.animationDuration = `${Math.random() * 12 + 8}s`; // Slower bubbles
        bubble.style.animationDelay = `${Math.random() * 4}s`;
        bubbleContainer.appendChild(bubble);
    }
}
document.addEventListener('DOMContentLoaded', createBubbles);
</script>
''', unsafe_allow_html=True)

# Layout changes to improve user experience
st.title("Rhythm Maker: AI-Powered Music Generation")

# Create layout with columns to organize content better
col1, col2 = st.columns(2)

# Section 1: Upload or generate music
with col1:
    st.header("Upload or Generate Music")
    uploaded_file = st.file_uploader("Choose a music file", type=["mp3", "wav"])
    if uploaded_file:
        st.audio(uploaded_file, format='audio/mp3')

# Section 2: AI-generated music section
with col2:
    st.header("AI-Generated Music")
    st.text("Use our AI model to generate music based on your inputs.")

    # Simulate a button to generate music
    if st.button('Generate AI Music'):
        with st.spinner('Generating music...'):
            st.success("AI Music Generated!")

# Adding footer information
st.markdown('''
<footer style='text-align: center; color: lightgray; font-size: 0.8em;'>
    Created by AI DJ © 2024. All rights reserved.
</footer>
''', unsafe_allow_html=True)



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

if st.button("Make New Song"):
    st.rerun()

