import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import streamlit as st
import wave

os.environ['HF_TOKEN'] = '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'

# Load CSS from index.html
with open('templates/index.html', 'r') as file:
    html_content = file.read()
    css_content = html_content.split('<style>')[1].split('</style>')[0]

st.set_page_config(page_title="Rhythm Maker", layout="wide")

# Apply custom CSS
st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

class AdvancedAIDJ:
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'], attn_implementation="eager").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'])

    def generate_song(self, style, duration=60):
        prompt = f"Create an engaging {style} song with a catchy melody and rhythm"
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        sampling_rate = 32000
        total_samples = duration * sampling_rate
        
        max_new_tokens = min(int(total_samples / 256), 1024)  # Limit to 1024 tokens
        
        audio_values = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.5,
            temperature=0.8
        )
        
        audio_data = audio_values[0].cpu().numpy()
        
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        self.last_generated_audio = audio_data
        self.last_sampling_rate = sampling_rate
        
        return audio_data, sampling_rate

    def export_song(self, style, filename="generated_song.wav"):
        audio_data = self.last_generated_audio
        sampling_rate = self.last_sampling_rate
        
        audio_data = np.int16(audio_data * 32767)

        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sampling_rate))
            wav_file.writeframes(audio_data.tobytes())

        return filename

ai_dj = AdvancedAIDJ()

# HTML structure from index.html
st.markdown("""
<div class="bubble-container"></div>
<h1>Rhythm Maker</h1>
""", unsafe_allow_html=True)

# Style buttons
styles = ["jazz", "rock", "electronic", "classical"]
cols = st.columns(len(styles))
for i, style in enumerate(styles):
    if cols[i].button(style.capitalize()):
        selected_style = style

st.markdown('<div id="analyzingText" style="display:none;"><p>AI is analyzing and creating your unique rhythm...</p></div>', unsafe_allow_html=True)

if 'selected_style' in locals():
    with st.spinner("Generating your song..."):
        audio_data, sampling_rate = ai_dj.generate_song(selected_style)
    
    st.audio(audio_data, format='audio/wav', sample_rate=sampling_rate)
    
    st.markdown("""
    <div id="downloadButton">
        <button onclick="location.href='#'">Download Song</button>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div>
    <button onclick="location.reload()">Make New Song</button>
</div>
""", unsafe_allow_html=True)

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
window.addEventListener('load', createBubbles);
</script>
""", unsafe_allow_html=True)