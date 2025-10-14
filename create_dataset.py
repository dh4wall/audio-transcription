import os
import subprocess

# Get the directory where the script is located
sd = os.path.dirname(os.path.abspath(__file__))

a1 = os.path.join(sd, "audiofiles_cleaned")
t1 = os.path.join(sd, "output")
d1 = os.path.join(sd, "My_Varhadi_TTS_Dataset")
w1 = os.path.join(d1, "wavs")
m1 = os.path.join(d1, "metadata.csv")
p1 = "varhadi"

def create_ds():
    print("🚀 Starting dataset creation...")

    os.makedirs(w1, exist_ok=True)
    print(f"Created dataset directory at: {d1}")

    try:
        # Get all .txt files and sort them numerically by filename
        txt_files_list = [f for f in os.listdir(t1) if f.endswith('.txt')]
        # Sorts '10.txt' after '9.txt' by converting the name to an integer
        sorted_txt_files = sorted(txt_files_list, key=lambda x: int(os.path.splitext(x)[0]))
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: Could not find or parse files in '{t1}'. Ensure it exists and contains correctly named .txt files (e.g., '0.txt', '1.txt'). Details: {e}")
        return

    m2 = []
    c1 = 1

    for txt_file in sorted_txt_files:
        # Get base name (e.g., '5' from '5.txt') to find matching audio
        base_name = os.path.splitext(txt_file)[0]
        s2 = os.path.join(t1, txt_file)
        s1 = os.path.join(a1, f"{base_name}_normalized.mp3")

        if os.path.exists(s1):
            n1 = f"{p1}_{c1:04d}.wav"
            d2 = os.path.join(w1, n1)

            print(f"Processing: {os.path.basename(s1)} & {txt_file} -> {n1}")
            cmd = [
                'ffmpeg',
                '-i', s1,
                '-ar', '22050',
                '-ac', '1',
                d2,
                '-hide_banner',
                '-loglevel', 'error'
            ]
            subprocess.run(cmd)

            with open(s2, 'r', encoding='utf-8') as f:
                tr = f.read().strip().replace('\n', ' ')

            if tr:
                m2.append(f"{n1}|{tr}")
                c1 += 1
        else:
            print(f"⚠️ Warning: Skipping {txt_file}. Could not find matching audio file '{base_name}.mp3' in '{a1}'.")

    print(f"\n✍️ Writing metadata for {len(m2)} files to {m1}...")
    with open(m1, 'w', encoding='utf-8') as f:
        for l in m2:
            f.write(l + '\n')
            
    print("\n✅ Dataset creation complete!")
    print(f"Your new dataset is ready in the '{d1}' folder.")

if __name__ == '__main__':
    create_ds()

