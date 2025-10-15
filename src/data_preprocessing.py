
"""
Data Preprocessing Script
- Loads raw audio files
- Trims silence
- Normalizes audio
- Saves processed audio for feature extraction
"""

import os
import librosa
import soundfile as sf
import argparse

def preprocess_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for f in audio_files:
        file_path = os.path.join(input_dir, f)
        y, sr = librosa.load(file_path, sr=None)
        
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y)
        
        # Normalize
        y_norm = librosa.util.normalize(y_trimmed)
        
        # Save processed audio
        output_path = os.path.join(output_dir, f)
        sf.write(output_path, y_norm, sr)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument("--input", type=str, required=True, help="Input folder with raw audio")
    parser.add_argument("--output", type=str, required=True, help="Output folder for processed audio")
    args = parser.parse_args()
    
    preprocess_audio(args.input, args.output)

