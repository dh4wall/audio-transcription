import os
import stable_whisper
import re
import torch

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def transcribe_audio(audio_path, output_file, model_name="large-v3"):
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at {audio_path}")
        return
        
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Loading the '{model_name}' Whisper model onto the {compute_device.upper()}...")
    
    try:
        a1 = stable_whisper.load_model(model_name, device=compute_device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print(f"🎤 Transcribing {audio_path} with high-accuracy settings...")
    
    try:
        t1 = a1.transcribe(
            audio_path,
            language='mr',
            beam_size=10,
            vad=True,
            no_speech_threshold=0.4,
            logprob_threshold=-1.0,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        )
    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")
        return

    print("🖋️ Cleaning and saving the full transcript...")
    full_transcript = t1.text
    cleaned_transcript = clean_text(full_transcript)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_transcript)
        print(f"✅ Success! Full transcript saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == '__main__':
    AUDIO_FILE_PATH = "audiofiles/4.mp3"
    OUTPUT_TRANSCRIPT_FILE = "output/4.txt"
    WHISPER_MODEL = "large-v3"

    transcribe_audio(
        audio_path=AUDIO_FILE_PATH,
        output_file=OUTPUT_TRANSCRIPT_FILE,
        model_name=WHISPER_MODEL
    )