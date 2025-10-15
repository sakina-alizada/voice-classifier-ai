
"""
Model Evaluation Script
- Loads trained model
- Predicts on test data
- Displays confusion matrix and accuracy
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def evaluate_model(features_file, model_file, label_encoder_file):
    # Load features
    df = pd.read_csv(features_file)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Load label encoder
    le = joblib.load(label_encoder_file)
    y_encoded = le.transform(y)
    
    # Load model
    model = tf.keras.models.load_model(model_file)
    
    # Predict
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_encoded, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Classification Report
    report = classification_report(y_encoded, y_pred_classes, target_names=le.classes_)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained audio classification model")
    parser.add_argument("--features", type=str, required=True, help="CSV file with features")
    parser.add_argument("--model", type=str, required=True, help="Trained model file")
    parser.add_argument("--le", type=str, required=True, help="Label encoder file")
    args = parser.parse_args()
    
    evaluate_model(args.features, args.model, args.le)

