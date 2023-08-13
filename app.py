import gradio as gr
from pydub.playback import play
from mido import MidiFile, MidiTrack, Message
import openai
import subprocess
from pydub import AudioSegment

openai.api_key = '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'
# soundfont_file = "sountrack.sf2"
rock_music_prompt = """
Generate a piece of rock music that embodies the characteristics of rock music. 
The music should have energetic electric guitars, a driving rhythm section, and powerful vocals. 
It should feature a moderate to fast tempo, guitar distortion, and dynamic shifts between quiet and loud sections. 
The lyrics should address themes commonly found in rock music, such as rebellion, love, and personal experiences.
"""

def convert_text_to_music(details):
    # prompt = f"Generate music based on the following text by adding more feelings: {details}"
    prompt = rock_music_prompt + f"\nUser input: {details}"
    response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100, 
            top_p=1
        )
    music_text = response.choices[0].text.strip()
    print(music_text)

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note_char in music_text:
        if note_char.isdigit():
            note = int(note_char)
            track.append(Message("note_on", note=note, velocity=64, time=0))
            track.append(Message("note_off", note=note, velocity=64, time=480))

    generated_music_file = "generated_music.mid"
    mid.save(generated_music_file)

    # wav_output_file = "generated_music.wav"
    # subprocess.run(["fluidsynth", "-F", wav_output_file, soundfont_file, generated_music_file])

    # # Load the generated WAV file
    # generated_wav = AudioSegment.from_wav(wav_output_file)

    # # Save the generated WAV file as MP3
    # generated_mp3_file = "generated_music.mp3"
    # generated_wav.export(generated_mp3_file, format="mp3")
    
    return generated_music_file

text_input = gr.Textbox(label="Enter Written Details", type="text")
music_output = gr.File(label="Generated Music", type="file")

gr.Interface(
    fn=convert_text_to_music,
    inputs=text_input,
    outputs=music_output
).launch()






