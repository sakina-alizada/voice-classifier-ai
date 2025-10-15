
"""
Unit test for feature_extraction.py
- Verifies that features are correctly extracted from audio files
"""

import unittest
import os
import numpy as np
import pandas as pd
from src.feature_extraction import extract_features, extract_features_from_folder

class TestFeatureExtraction(unittest.TestCase):
    
    def setUp(self):
        # Sample audio folder (small demo dataset)
        self.input_dir = '../data/sample_audio/'
        self.output_file = '../data/test_features.csv'
        os.makedirs(self.input_dir, exist_ok=True)

        # Optional: create a dummy audio file if folder is empty
        if len(os.listdir(self.input_dir)) == 0:
            import soundfile as sf
            import numpy as np
            sr = 22050
            y = np.random.uniform(-1, 1, sr*2)  # 2 seconds random audio
            sf.write(os.path.join(self.input_dir, 'dummy_happy.wav'), y, sr)

    def test_extract_features_shape(self):
        # Take first file and extract features
        audio_files = [f for f in os.listdir(self.input_dir) if f.endswith('.wav')]
        features = extract_features(os.path.join(self.input_dir, audio_files[0]))
        
        # Feature vector should have expected length (13 MFCC + 12 Chroma + 7 Contrast + 6 Tonnetz = 38)
        self.assertEqual(len(features), 38, "Feature vector length should be 38")

    def test_extract_features_from_folder(self):
        # Extract features for folder
        extract_features_from_folder(self.input_dir, self.output_file)
        
        # Check output CSV exists
        self.assertTrue(os.path.exists(self.output_file), "Output CSV should exist")
        
        # Load CSV and check shape
        df = pd.read_csv(self.output_file)
        self.assertTrue('label' in df.columns, "CSV should contain 'label' column")
        self.assertGreater(df.shape[0], 0, "CSV should have at least one row")
        self.assertEqual(df.shape[1], 39, "CSV should have 38 features + 1 label column")
    
    def tearDown(self):
        # Remove test CSV
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

if __name__ == "__main__":
    unittest.main()

