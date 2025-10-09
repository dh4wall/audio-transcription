import os
import subprocess

input_dir = "audiofiles"
output_dir = "audiofiles_cleaned"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through numbers from 5 to 39
for i in range(5, 40):
    filename = f"{i}.mp3"
    input_path = os.path.join(input_dir, filename)
    
    output_filename = f"{i}_normalized.mp3"
    output_path = os.path.join(output_dir, output_filename)

    # Check if the input file exists
    if os.path.exists(input_path):
        print(f"Normalizing {input_path}...")
        try:
            # Construct and run the ffmpeg command
            command = [
                'ffmpeg',
                '-i', input_path,
                '-af', 'loudnorm',
                output_path,
                '-hide_banner',
                '-loglevel', 'error'
            ]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to process {filename}. Error: {e}")
    else:
        print(f"Warning: File {input_path} not found, skipping.")

print("✅ All files have been processed!")