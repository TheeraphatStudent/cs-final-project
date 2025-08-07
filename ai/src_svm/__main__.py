import os
import sys
from model import URLClassifierSVM
from data_processes import DataProcessorSVM
from measurement import ModelPerformanceSVM
import numpy as np
import json
from __types__.name import NameManagement

def display_svm_performance_metrics():
    """
    Display enhanced SVM model performance metrics
    
    Features:
    1. Comprehensive metric loading from JSON
    2. Advanced performance visualization
    3. Feature importance analysis
    4. Model comparison capabilities
    """
    try:
        metrics_path = 'metrics/svm_model_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            print("\n" + "="*60)
            print("SVM Model Performance Metrics")
            print("="*60)
            
            # Display basic metrics
            if 'accuracy' in metrics:
                print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
            if 'precision' in metrics:
                print(f"Precision:    {metrics['precision']:.4f} ({metrics['precision']:.2%})")
            if 'recall' in metrics:
                print(f"Recall:       {metrics['recall']:.4f} ({metrics['recall']:.2%})")
            if 'f1' in metrics:
                print(f"F1 Score:     {metrics['f1']:.4f} ({metrics['f1']:.2%})")
            
            # Display advanced metrics
            if 'roc_auc' in metrics:
                print(f"ROC AUC:      {metrics['roc_auc']:.4f}")
            if 'average_precision' in metrics:
                print(f"Avg Precision: {metrics['average_precision']:.4f}")
            if 'specificity' in metrics:
                print(f"Specificity:  {metrics['specificity']:.4f}")
            if 'sensitivity' in metrics:
                print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
            
            # Display confusion matrix
            if 'confusion_matrix' in metrics:
                print("\nConfusion Matrix:")
                cm = metrics['confusion_matrix']
                print(f"                Predicted")
                print(f"              Safe  Malicious")
                print(f"Actual Safe    {cm[0][0]:>4}    {cm[0][1]:>4}")
                print(f"Actual Malicious {cm[1][0]:>4}    {cm[1][1]:>4}")
                
                # Calculate additional metrics
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                total = tn + fp + fn + tp
                print(f"\nDetailed Analysis:")
                print(f"True Negatives:  {tn} ({tn/total:.2%})")
                print(f"False Positives: {fp} ({fp/total:.2%})")
                print(f"False Negatives: {fn} ({fn/total:.2%})")
                print(f"True Positives:  {tp} ({tp/total:.2%})")
            
            # Display classification report if available
            if 'classification_report' in metrics:
                print("\nDetailed Classification Report:")
                report = metrics['classification_report']
                if 'Safe' in report:
                    safe_metrics = report['Safe']
                    print(f"Safe URLs:")
                    print(f"  Precision: {safe_metrics['precision']:.4f}")
                    print(f"  Recall:    {safe_metrics['recall']:.4f}")
                    print(f"  F1-Score:  {safe_metrics['f1-score']:.4f}")
                    print(f"  Support:   {safe_metrics['support']}")
                
                if 'Malicious' in report:
                    malicious_metrics = report['Malicious']
                    print(f"Malicious URLs:")
                    print(f"  Precision: {malicious_metrics['precision']:.4f}")
                    print(f"  Recall:    {malicious_metrics['recall']:.4f}")
                    print(f"  F1-Score:  {malicious_metrics['f1-score']:.4f}")
                    print(f"  Support:   {malicious_metrics['support']}")
            
            print("="*60)
        else:
            print("No SVM performance metrics available.")
            print("Please train the model first.")
    except Exception as e:
        print(f"\nError displaying metrics: {str(e)}")

def display_feature_importance():
    """
    Display feature importance analysis
    """
    try:
        classifier = URLClassifierSVM()
        if os.path.exists(classifier.model_path):
            classifier.load_model()
            importance = classifier.get_feature_importance()
            
            if importance:
                print("\n" + "="*60)
                print("Feature Importance Analysis")
                print("="*60)
                
                # Display top 15 features
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
                
                print(f"{'Feature':<25} {'Importance':<15}")
                print("-" * 40)
                for feature, importance_score in top_features:
                    print(f"{feature:<25} {importance_score:<15.4f}")
                
                print("="*60)
            else:
                print("Feature importance not available for this model type.")
        else:
            print("No trained model found. Please train the model first.")
    except Exception as e:
        print(f"Error displaying feature importance: {str(e)}")

def compare_models():
    """
    Compare SVM model with Neural Network model
    """
    try:
        # Load SVM metrics
        svm_metrics_path = 'metrics/svm_model_metrics.json'
        nn_metrics_path = 'metrics/model_metrics.json'
        
        if os.path.exists(svm_metrics_path) and os.path.exists(nn_metrics_path):
            with open(svm_metrics_path, 'r') as f:
                svm_metrics = json.load(f)
            with open(nn_metrics_path, 'r') as f:
                nn_metrics = json.load(f)
            
            print("\n" + "="*60)
            print("Model Comparison: SVM vs Neural Network")
            print("="*60)
            
            comparison_metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            print(f"{'Metric':<15} {'SVM':<10} {'Neural Net':<12} {'Difference':<12}")
            print("-" * 60)
            
            for metric in comparison_metrics:
                if metric in svm_metrics and metric in nn_metrics:
                    svm_val = svm_metrics[metric]
                    nn_val = nn_metrics[metric]
                    diff = svm_val - nn_val
                    diff_sign = "+" if diff >= 0 else ""
                    
                    print(f"{metric.capitalize():<15} {svm_val:<10.4f} {nn_val:<12.4f} {diff_sign}{diff:<11.4f}")
            
            # Additional comparison metrics
            if 'roc_auc' in svm_metrics and 'roc_auc' in nn_metrics:
                svm_auc = svm_metrics['roc_auc']
                nn_auc = nn_metrics.get('roc_auc', 0)
                auc_diff = svm_auc - nn_auc
                auc_sign = "+" if auc_diff >= 0 else ""
                print(f"{'ROC AUC':<15} {svm_auc:<10.4f} {nn_auc:<12.4f} {auc_sign}{auc_diff:<11.4f}")
            
            print("="*60)
            
            # Determine winner
            svm_wins = 0
            nn_wins = 0
            
            for metric in comparison_metrics:
                if metric in svm_metrics and metric in nn_metrics:
                    if svm_metrics[metric] > nn_metrics[metric]:
                        svm_wins += 1
                    elif nn_metrics[metric] > svm_metrics[metric]:
                        nn_wins += 1
            
            print(f"\nOverall Performance:")
            print(f"SVM wins: {svm_wins} metrics")
            print(f"Neural Network wins: {nn_wins} metrics")
            
            if svm_wins > nn_wins:
                print("SVM performs better overall!")
            elif nn_wins > svm_wins:
                print("Neural Network performs better overall!")
            else:
                print("Both models perform similarly!")
                
        else:
            print("Both model metrics files not found. Please train both models first.")
    except Exception as e:
        print(f"Error comparing models: {str(e)}")

def main():
    os.makedirs('dataset/processed', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize components
    processor = DataProcessorSVM(
        input_path='dataset/processed/data_cleanup.csv',
        output_path='dataset/processed/processed_data_svm.csv'
    )
    
    performance = ModelPerformanceSVM(model_name='svm_url_classifier')
    classifier = URLClassifierSVM()
    name_manager = NameManagement()
    model_exists = os.path.exists(f'{name_manager.getFileName()}_svm.joblib')
    
    if not model_exists:
        print("No existing SVM model found. Training new model...")
        try:
            # Process data first
            print("Processing dataset with enhanced features...")
            df = processor.process_data()
            if df is None:
                print("Error processing data. Exiting...")
                sys.exit(1)
            
            # Get data summary
            summary = processor.get_data_summary()
            if summary:
                print(f"\nDataset Summary:")
                print(f"Total URLs: {summary['total_urls']}")
                print(f"Malicious: {summary['malicious_count']} ({summary['malicious_ratio']:.2%})")
                print(f"Safe: {summary['safe_count']} ({1-summary['malicious_ratio']:.2%})")
                print(f"Average URL length: {summary['avg_url_length']:.1f}")
                print(f"Average entropy: {summary['avg_entropy']:.2f}")
            
            # Train the SVM model
            print("\nTraining SVM model with hyperparameter optimization...")
            metrics = classifier.train('dataset/processed/processed_data_svm.csv')
            
            # Update performance metrics
            performance.update_history({
                'hyperparameters': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            })
            
            # Save metrics and generate plots
            performance.save_metrics('metrics/svm_model_metrics.json')
            classifier.visualize_model(save_path='plots/')
            
            print("SVM model training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            sys.exit(1)
    else:
        print("Loading existing SVM model...")
        try:
            classifier.load_model()
            # Load existing metrics if available
            if os.path.exists('metrics/svm_model_metrics.json'):
                performance.load_metrics('metrics/svm_model_metrics.json')
            print("SVM model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    # Main program loop
    while True:
        print("\n" + "="*60)
        print("Enhanced URL Malicious Detection System (SVM)")
        print("="*60)
        print("1. Check URL")
        print("2. Train/Retrain SVM model")
        print("3. Model performance analysis")
        # print("4. Feature importance analysis")
        # print("5. Compare with Neural Network")
        # print("6. Visualize model performance")
        # print("7. Data analysis")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            url = input("Enter URL to check: ")
            try:
                prediction, confidence, reason = classifier.predict(url)
                print("\n" + "-"*50)
                print("Prediction Results:")
                print("-"*50)
                print(f"URL: {url}")
                print(f"Prediction: {prediction.upper()}")
                print(f"Confidence: {confidence:.2%}")
                print(f"Reason: {reason}")
                print("-"*50)
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                
        elif choice == '2':
            print("Training/Retraining SVM model with enhanced features...")
            try:
                # Process data first
                print("Processing dataset...")
                df = processor.process_data()
                if df is None:
                    print("Error processing data. Exiting...")
                    continue
                    
                # Train the model
                metrics = classifier.train('dataset/processed/processed_data_svm.csv')
                
                # Update performance metrics
                performance.update_history({
                    'hyperparameters': {
                        'kernel': 'rbf',
                        'C': 1.0,
                        'gamma': 'scale'
                    }
                })
                
                # Save metrics and generate plots
                performance.save_metrics('metrics/svm_model_metrics.json')
                classifier.visualize_model(save_path='plots/')
                
                print("SVM model retrained successfully!")
            except Exception as e:
                print(f"Error during retraining: {str(e)}")
                
        elif choice == '3':
            display_svm_performance_metrics()
            
        elif choice == '4':
            display_feature_importance()
            
        elif choice == '5':
            compare_models()
            
        elif choice == '6':
            try:
                classifier.visualize_model(save_path='plots/')
                print("Model visualizations have been saved to 'plots/' directory")
                print("Generated files:")
                print("- svm_confusion_matrix.png")
                print("- svm_feature_importance.png")
            except Exception as e:
                print(f"Error visualizing model: {str(e)}")
            
        elif choice == '7':
            try:
                print("\nData Analysis:")
                summary = processor.get_data_summary()
                if summary:
                    print(f"Dataset Summary:")
                    print(f"Total URLs: {summary['total_urls']}")
                    print(f"Malicious: {summary['malicious_count']} ({summary['malicious_ratio']:.2%})")
                    print(f"Safe: {summary['safe_count']} ({1-summary['malicious_ratio']:.2%})")
                    print(f"Average URL length: {summary['avg_url_length']:.1f}")
                    print(f"Average entropy: {summary['avg_entropy']:.2f}")
                    print(f"HTTPS ratio: {summary['https_ratio']:.2%}")
                    print(f"Suspicious extension ratio: {summary['suspicious_extension_ratio']:.2%}")
                    print(f"Suspicious keyword ratio: {summary['suspicious_keyword_ratio']:.2%}")
                    print(f"Legitimate domain ratio: {summary['legitimate_domain_ratio']:.2%}")
                
                validation = processor.validate_data()
                if validation:
                    print(f"\nData Validation:")
                    print(f"Missing values: {sum(validation['missing_values'].values())}")
                    print(f"Duplicate URLs: {validation['duplicate_urls']}")
                    print(f"Invalid URLs: {validation['invalid_urls']}")
            except Exception as e:
                print(f"Error in data analysis: {str(e)}")
            
        elif choice == '8':
            print("Exiting enhanced SVM URL classification system...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 