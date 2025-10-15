
"""
Feature Extraction Script
- Extracts MFCC, Chroma, Spectral Contrast, Tonnetz
- Saves features to CSV
"""

import os
import librosa
import numpy as np
import pandas as pd
import argparse

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    
    # Tonnetz
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    
    return np.concatenate([mfccs_mean, chroma_mean, contrast_mean, tonnetz_mean])

def extract_features_from_folder(input_dir, output_file):
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    features_list = []
    labels = []

    for f in audio_files:
        file_path = os.path.join(input_dir, f)
        label = f.split('_')[0]  # assuming filenames like happy_01.wav
        features = extract_features(file_path)
        features_list.append(features)
        labels.append(label)

    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    df['label'] = labels
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("--input", type=str, required=True, help="Folder with processed audio")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for features")
    args = parser.parse_args()
    
    extract_features_from_folder(args.input, args.output)

