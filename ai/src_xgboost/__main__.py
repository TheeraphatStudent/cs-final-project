import os
import sys
from model import URLClassifier
from data_processes import DataProcessor
from measurement import ModelPerformance
import numpy as np
import json
from __types__.name import NameManagement
import pandas as pd

def display_performance_metrics():
    try:
        metrics_path = 'metrics/model_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            print("\nModel Performance Metrics:")
            
            if 'accuracy' in metrics:
                print(f"Accuracy: {metrics['accuracy']:.2%}")
            if 'precision' in metrics:
                print(f"Precision: {metrics['precision']:.2%}")
            if 'recall' in metrics:
                print(f"Recall: {metrics['recall']:.2%}")
            if 'f1' in metrics:
                print(f"F1 Score: {metrics['f1']:.2%}")
            
            if 'confusion_matrix' in metrics:
                print("\nConfusion Matrix:")
                cm = metrics['confusion_matrix']
                print(f"True Negatives: {cm[0][0]}")
                print(f"False Positives: {cm[0][1]}")
                print(f"False Negatives: {cm[1][0]}")
                print(f"True Positives: {cm[1][1]}")
            
            if 'training_history' in metrics:
                print("\nTraining History:")
                history = metrics['training_history']
                if 'accuracy' in history and history['accuracy']:
                    print(f"Final Training Accuracy: {history['accuracy'][-1]:.2%}")
                if 'val_accuracy' in history and history['val_accuracy']:
                    print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.2%}")
        else:
            print("No performance metrics available.")
    except Exception as e:
        print(f"\nError displaying metrics: {str(e)}")

def main():
    os.makedirs('dataset/processed', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    processor = DataProcessor(
        input_path='./dataset/processed/data_cleanup.csv',
        output_path='dataset/processed/processed_data.csv'
    )
    
    performance = ModelPerformance(model_name='url_classifier')
    classifier = URLClassifier()
    name_manager = NameManagement()
    model_exists = os.path.exists(f'{name_manager.getFileName()}.pkl')
    csv_exists = os.path.exists('dataset/processed/processed_data.csv')
    raw_data_exists = os.path.exists('dataset/processed/data_cleanup.csv')
    
    if not model_exists:
        print("No existing model found. Training new model...")
        try:
            if csv_exists:
                print("Using existing processed CSV file...")
                df = pd.read_csv('dataset/processed/processed_data.csv')
                print(f"Loaded {len(df)} records from existing CSV")
            elif raw_data_exists:
                print("Processing raw dataset...")
                df = processor.process_data()
                if df is None:
                    print("Error processing data. Exiting...")
                    sys.exit(1)
            else:
                print("No dataset found. Please ensure data_cleanup.csv exists in dataset/processed/")
                sys.exit(1)
                
            history = classifier.train('dataset/processed/processed_data.csv')
            
            performance.update_history(history)
            
            performance.save_metrics('metrics/model_metrics.json')
            performance.plot_training_history(save_path='plots/training_history.png')
            
            print("Model training completed successfully!")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            sys.exit(1)
    else:
        print("Loading existing model...")
        try:
            classifier.load_model()
            if os.path.exists('metrics/model_metrics.json'):
                performance.load_metrics('metrics/model_metrics.json')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    while True:
        print("\nURL Malicious Detection System")
        print("1. Check URL")
        print("2. Train model")
        print("3. Model performance")
        print("4. Visualize model")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            url = input("Enter URL to check: ")
            try:
                prediction, confidence, reason = classifier.predict(url)
                print("\nPrediction Results:")
                print(f"URL: {url}")
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.2%}")
                print(f"Reason: {reason}")
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                
        elif choice == '2':
            print("Training model with new data...")
            try:
                if csv_exists:
                    print("Using existing processed CSV file...")
                    df = pd.read_csv('dataset/processed/processed_data.csv')
                    print(f"Loaded {len(df)} records from existing CSV")
                elif raw_data_exists:
                    print("Processing raw dataset...")
                    df = processor.process_data()
                    if df is None:
                        print("Error processing data. Exiting...")
                        continue
                else:
                    print("No dataset found. Please ensure data_cleanup.csv exists in dataset/processed/")
                    continue
                    
                history = classifier.train('dataset/processed/processed_data.csv')
                
                performance.update_history(history)
                
                performance.save_metrics('metrics/model_metrics.json')
                performance.plot_training_history(save_path='plots/training_history.png')
                
                print("Model retrained successfully!")
            except Exception as e:
                print(f"Error during retraining: {str(e)}")
                
        elif choice == '3':
            display_performance_metrics()
            
        elif choice == '4':
            try:
                classifier.visualize_model()
                print("Model feature importance visualization has been saved to 'plots/feature_importance.png'")
            except Exception as e:
                print(f"Error visualizing model: {str(e)}")
            
        elif choice == '5':
            print("Exiting program...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()