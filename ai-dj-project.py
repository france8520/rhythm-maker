import os
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import wave
import struct
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context

os.environ['HF_TOKEN'] = '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'

app = Flask(__name__)

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
        
        for i in range(0, 101, 2):
            yield f"data:{i}\n\n"
            if i % 10 == 0:
                print(f"Generation progress: {i}%")
        
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    style = request.args.get('style', 'jazz')
    filename = f"{style}_song.wav"

    def generate_stream():
        for progress in ai_dj.generate_song(style):
            yield progress
        yield f"data:100\n\n"
        exported_filename = ai_dj.export_song(style, filename)
        yield f"data:DONE:{exported_filename}\n\n"

    return Response(stream_with_context(generate_stream()), content_type='text/event-stream')

@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
