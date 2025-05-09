import os
import sys
from model import URLClassifier
from data_processes import DataProcessor
from measurement import ModelPerformance
import numpy as np

def main():
    # Create necessary directories
    os.makedirs('dataset/processed', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data processor
    processor = DataProcessor(
        input_path='dataset/dataset_1.xlsx',
        output_path='dataset/processed/processed_data.csv'
    )
    
    # Initialize performance tracker
    performance = ModelPerformance(model_name='url_classifier')
    
    # Initialize the URL classifier
    classifier = URLClassifier()
    
    # Check if model exists
    model_exists = os.path.exists('my_model.keras')
    
    if not model_exists:
        print("No existing model found. Training new model...")
        try:
            # Process data first
            print("Processing dataset...")
            df = processor.process_data()
            if df is None:
                print("Error processing data. Exiting...")
                sys.exit(1)
                
            # Train the model
            history = classifier.train('dataset/processed/processed_data.csv')
            
            # Update performance metrics
            performance.update_history(history)
            
            # Save metrics and generate plots
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
            # Load existing metrics if available
            if os.path.exists('metrics/model_metrics.json'):
                performance.load_metrics('metrics/model_metrics.json')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    # Main program loop
    while True:
        print("\nURL Malicious Detection System")
        print("1. Check URL")
        print("2. Train model with new data")
        print("3. View model performance")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            url = input("Enter URL to check: ")
            try:
                result = classifier.predict(url)
                print("\nPrediction Results:")
                print(f"URL: {result['url']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                if 'reason' in result:
                    print(f"Reason: {result['reason']}")
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                
        elif choice == '2':
            print("Training model with new data...")
            try:
                # Process data first
                print("Processing dataset...")
                df = processor.process_data()
                if df is None:
                    print("Error processing data. Exiting...")
                    continue
                    
                # Train the model
                history = classifier.train('dataset/processed/processed_data.csv')
                
                # Update performance metrics
                performance.update_history(history)
                
                # Save metrics and generate plots
                performance.save_metrics('metrics/model_metrics.json')
                performance.plot_training_history(save_path='plots/training_history.png')
                
                print("Model retrained successfully!")
            except Exception as e:
                print(f"Error during retraining: {str(e)}")
                
        elif choice == '3':
            print("\nModel Performance Metrics:")
            latest_metrics = performance.get_latest_metrics()
            if latest_metrics:
                print(f"Accuracy: {latest_metrics['accuracy']:.2%}")
                print(f"Precision: {latest_metrics['precision']:.2%}")
                print(f"Recall: {latest_metrics['recall']:.2%}")
                print(f"F1 Score: {latest_metrics['f1']:.2%}")
                
                # Plot confusion matrix
                performance.plot_confusion_matrix(
                    np.array(latest_metrics['confusion_matrix']),
                    save_path='plots/confusion_matrix.png'
                )
                
                # Plot ROC curve if available
                if 'roc_curve' in latest_metrics:
                    performance.plot_roc_curve(
                        np.array(latest_metrics['roc_curve']['fpr']),
                        np.array(latest_metrics['roc_curve']['tpr']),
                        latest_metrics['roc_auc'],
                        save_path='plots/roc_curve.png'
                    )
                
                print("\nPlots have been saved to the 'plots' directory.")
            else:
                print("No performance metrics available.")
                
        elif choice == '4':
            print("Exiting program...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()