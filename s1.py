import os
import sys
import stable_whisper
import re
import torch
import gc

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def transcribe_single_file(filename, model_name="large"):
    input_dir = "audiofiles"
    output_dir = "output2" # <--- This is the change
    
    input_path = os.path.join(input_dir, filename)
    if not os.path.exists(input_path):
        print(f"❌ Error: Audio file not found at '{input_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Loading the '{model_name}' Whisper model onto the {compute_device.upper()}...")
    
    a1 = None
    try:
        a1 = stable_whisper.load_model(model_name, device=compute_device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print(f"🎤 Transcribing {filename}...")
    
    t1 = None
    try:
        t1 = a1.transcribe(
            input_path,
            language='mr',
            beam_size=10,
            vad=True,
            no_speech_threshold=0.4,
            logprob_threshold=-1.0,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        )

        print(f"🖋️ Cleaning and saving transcript to {output_path}...")
        full_transcript = t1.text
        cleaned_transcript = clean_text(full_transcript)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_transcript)
        print(f"✅ Success! Saved {output_filename}")

    except Exception as e:
        print(f"❌ Failed to process {filename}. Error: {e}")
    finally:
        a1 = t1 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python start.py <filename>")
        print("Example: python start.py 4.mp3")
        sys.exit(1)
        
    audio_filename = sys.argv[1]
    WHISPER_MODEL = "large"

    transcribe_single_file(
        filename=audio_filename,
        model_name=WHISPER_MODEL
    )