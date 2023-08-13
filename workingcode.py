import gradio as gr
import torch
import librosa
import gc
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.to(device)

# Define the ASR function
def convert_speech_to_text(audio):
    # Convert audio to WAV format
    audio = AudioSegment.from_mp3(audio)
    wav_audio_file = 'user_audio.wav'
    audio.export(wav_audio_file, format="wav")

    # Load and process audio
    input_audio, sr = librosa.load(wav_audio_file, sr=16000)
    
    # Tokenize
    input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values.to(device)
    
    # Get logits
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    
    return transcription[0]

# Create the Gradio interface
audio_input = gr.Audio(label="Upload an Audio", type="filepath")
text_output = gr.Textbox(label="Transcribed Text", type="text")

gr.Interface(fn=convert_speech_to_text, inputs=audio_input, outputs=text_output).launch()
