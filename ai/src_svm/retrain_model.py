#!/usr/bin/env python3
"""
Retrain SVM model with processed data and enhanced security features
"""

import pandas as pd
import numpy as np
from model import URLClassifierSVM
import os

def retrain_svm_model():
    """Retrain the SVM model with processed data and enhanced features"""
    
    print("🔄 Starting SVM model retraining...")
    
    # Check if processed data exists
    processed_data_path = 'dataset/processed/processed_data_svm.csv'
    if not os.path.exists(processed_data_path):
        print("❌ Processed data not found. Please run data processing first.")
        return
    
    # Load processed data
    print("📊 Loading processed data...")
    df = pd.read_csv(processed_data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check target distribution
    if 'is_malicious' in df.columns:
        target_col = 'is_malicious'
    else:
        target_col = 'isMalicious'
    
    print(f"Target distribution: {df[target_col].value_counts().to_dict()}")
    
    # Create classifier
    classifier = URLClassifierSVM()
    
    # Train the model
    print("🤖 Training SVM model...")
    try:
        metrics = classifier.train(processed_data_path)
        
        print("✅ Model training completed!")
        print(f"📈 Performance Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.2%}")
        print(f"   Precision: {metrics['precision']:.2%}")
        print(f"   Recall:    {metrics['recall']:.2%}")
        print(f"   F1-Score:  {metrics['f1']:.2%}")
        
        # Test the model with problematic URLs
        print("\n🧪 Testing model with security-critical URLs...")
        test_urls = [
            'www.google.com/folder/files.exe',
            'https://google.com/sample/file.exe',
            'www.google.com/sample/exh.exe', 
            'https://google.com',
            'https://github.com/user/project',
            'http://suspicious-site.com/login.exe'
        ]
        
        for url in test_urls:
            result = classifier.predict(url)
            status = "✅" if ((".exe" in url and result[0] == 'malicious') or 
                           (".exe" not in url and result[0] == 'safe')) else "❌"
            print(f"{status} {url}: {result[0].upper()} ({result[1]:.1%})")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return None

if __name__ == "__main__":
    retrain_svm_model() 