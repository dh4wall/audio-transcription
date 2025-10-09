import os
import subprocess

source_audio_dir = "audiofiles_cleaned"
source_txt_dir = "output"
dataset_dir = "My_Varhadi_TTS_Dataset"
wavs_dir = os.path.join(dataset_dir, "wavs")
metadata_path = os.path.join(dataset_dir, "metadata.csv")
file_prefix = "varhadi"

def create_dataset():
    print("🚀 Starting dataset creation...")

    os.makedirs(wavs_dir, exist_ok=True)
    print(f"Created dataset directory at: {dataset_dir}")

    try:
        txt_files = sorted([f for f in os.listdir(source_txt_dir) if f.endswith('.txt')])
        audio_files_map = {os.path.splitext(f)[0]: f for f in os.listdir(source_audio_dir)}
    except FileNotFoundError as e:
        print(f"❌ Error: Make sure the '{source_audio_dir}' and '{source_txt_dir}' folders exist. Details: {e}")
        return

    metadata = []
    counter = 1

    for txt_file in txt_files:
        txt_base_name = os.path.splitext(txt_file)[0]  # e.g., "5"

        # Corrected logic to find a matching audio file
        matching_audio_file = None
        for audio_base_name, audio_full_filename in audio_files_map.items():
            if audio_base_name.startswith(txt_base_name + '_') or audio_base_name == txt_base_name:
                matching_audio_file = audio_full_filename
                break

        if matching_audio_file:
            source_audio_path = os.path.join(source_audio_dir, matching_audio_file)
            source_txt_path = os.path.join(source_txt_dir, txt_file)

            new_wav_filename = f"{file_prefix}_{counter:04d}.wav"
            dest_wav_path = os.path.join(wavs_dir, new_wav_filename)

            print(f"Processing: {matching_audio_file} -> {new_wav_filename}")
            command = [
                'ffmpeg',
                '-i', source_audio_path,
                '-ar', '22050',
                '-ac', '1',
                dest_wav_path,
                '-hide_banner',
                '-loglevel', 'error'
            ]
            subprocess.run(command)

            with open(source_txt_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip().replace('\n', ' ')

            if transcript:
                metadata.append(f"{new_wav_filename}|{transcript}")
                counter += 1
        else:
            print(f"⚠️ Warning: No matching audio file found for {txt_file}, skipping.")

    print(f"\n✍️ Writing metadata for {len(metadata)} files to {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for line in metadata:
            f.write(line + '\n')
            
    print("\n✅ Dataset creation complete!")
    print(f"Your new dataset is ready in the '{dataset_dir}' folder.")

if __name__ == '__main__':
    create_dataset()