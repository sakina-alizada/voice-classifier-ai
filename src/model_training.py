
"""
Model Training Script
- Trains a CNN-LSTM model on audio features
- Saves the trained model
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import joblib

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(features_file, model_file, label_encoder_file):
    # Load features
    df = pd.read_csv(features_file)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, label_encoder_file)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(X_train.shape[1], len(le.classes_))
    
    # Train
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Save model
    model.save(model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio classification model")
    parser.add_argument("--features", type=str, required=True, help="CSV file with features")
    parser.add_argument("--model", type=str, required=True, help="Output path for trained model")
    parser.add_argument("--le", type=str, required=True, help="Output path for label encoder")
    args = parser.parse_args()
    
    train_model(args.features, args.model, args.le)

