import streamlit as st
import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np

# Set page config
st.set_page_config(page_title="Rhythm Maker", layout="wide")

# Apply custom CSS
def apply_custom_css():
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
    }
    .stButton > button {
        color: #4B0082;
        background-color: white;
        font-weight: bold;
        padding: 10px 15px;
        font-size: 16px;
    }
    #style-buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'], attn_implementation="eager").to(device)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", token=os.environ['HF_TOKEN'])
    return model, processor, device

# Generate song
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

# Main app
def main():
    apply_custom_css()
    
    st.title("Rhythm Maker")
    st.write("Welcome to the AI DJ Project! Generate your own music with AI.")
    
    # Initialize session state
    if 'selected_style' not in st.session_state:
        st.session_state.selected_style = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'sampling_rate' not in st.session_state:
        st.session_state.sampling_rate = None
    
    # Style selection buttons
    st.markdown('<div id="style-buttons">', unsafe_allow_html=True)
    for style in ["jazz", "rock", "electronic", "classical"]:
        if st.button(style.capitalize()):
            st.session_state.selected_style = style
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Duration slider
    duration = st.slider("Select song duration (seconds)", 10, 120, 60)
    
    # Generate button
    if st.button("Generate Song"):
        if st.session_state.selected_style:
            with st.spinner(f"Generating your {st.session_state.selected_style} song..."):
                try:
                    st.session_state.audio_data, st.session_state.sampling_rate = generate_song(st.session_state.selected_style, duration)
                    st.success("Song generated successfully!")
                except Exception as e:
                    st.error(f"An error occurred while generating the song: {str(e)}")
        else:
            st.warning("Please select a music style first.")
    
    # Display audio player and download button if audio is generated
    if st.session_state.audio_data is not None and st.session_state.sampling_rate is not None:
        st.audio(st.session_state.audio_data, format='audio/wav', sample_rate=st.session_state.sampling_rate)
        
        st.download_button(
            label="Download Song",
            data=st.session_state.audio_data.tobytes(),
            file_name=f"{st.session_state.selected_style}_song.wav",
            mime="audio/wav"
        )

if __name__ == "__main__":
    os.environ['HF_TOKEN'] = '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'
    model, processor, device = load_model()
    main()