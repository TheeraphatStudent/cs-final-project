import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import json
import os

class ModelPerformance:
    """
    Model Performance Tracker and Visualizer
    
    
    - Tracks model performance metrics during training
    - Generates visualizations for model evaluation
    - Manages metric storage and retrieval
    - Implements comprehensive performance analysis
    """

    def __init__(self, model_name):
        """
        Initialize the ModelPerformance tracker
        
        
        - Sets up metric tracking dictionaries
        - Initializes history storage for training metrics
        - Prepares for performance visualization
        """
        self.model_name = model_name
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        self.metrics_history = []
        
    def update_history(self, history):
        """
        Update training history with new metrics
        
        
        1. Metric Tracking:
           - Store training metrics
           - Track validation metrics
           - Update performance history
        2. Data Management:
           - Append new metrics to history
           - Maintain chronological order
        """
        for key in self.history:
            if key in history:
                self.history[key].append(history[key])
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate various performance metrics
        
        
        1. Classification Metrics:
           - Accuracy: Overall prediction accuracy
           - Precision: True positive rate among positive predictions
           - Recall: True positive rate among actual positives
           - F1 Score: Harmonic mean of precision and recall
        2. Confusion Matrix:
           - True Positives (TP)
           - False Positives (FP)
           - True Negatives (TN)
           - False Negatives (FN)
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history metrics
        
        
        1. Accuracy Plot:
           - Training accuracy over epochs
           - Validation accuracy over epochs
        2. Loss Plot:
           - Training loss over epochs
           - Validation loss over epochs
        3. Visualization:
           - Subplot creation
           - Axis labeling
           - Legend placement
        """
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        
        1. Matrix Visualization:
           - Create heatmap using seaborn
           - Add annotations for each cell
        2. Formatting:
           - Color scheme selection
           - Axis labels
           - Title placement
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """
        Plot ROC curve
        
        
        1. Curve Plotting:
           - Plot ROC curve with AUC score
           - Add diagonal reference line
        2. Formatting:
           - Axis limits
           - Labels and title
           - Legend placement
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_metrics(self, filepath):
        """
        Save metrics to a JSON file
        
        
        1. Data Serialization:
           - Convert metrics to JSON format
           - Include model name and history
        2. File Management:
           - Create directory if needed
           - Write to JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'model_name': self.model_name,
            'history': self.history,
            'metrics_history': self.metrics_history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load_metrics(self, filepath):
        """
        Load metrics from a JSON file
        
        
        1. File Reading:
           - Read JSON file
           - Parse metrics data
        2. Data Restoration:
           - Update model name
           - Restore history
           - Load metrics history
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.model_name = data.get('model_name', self.model_name)
                self.history = data.get('history', self.history)
                self.metrics_history = data.get('metrics_history', self.metrics_history)
    
    def get_latest_metrics(self):
        """
        Get the latest performance metrics
        
        
        - Retrieve most recent metrics from history
        - Return None if no metrics available
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report
        
        
        1. Report Generation:
           - Calculate precision, recall, F1-score
           - Generate per-class metrics
        2. Output Formatting:
           - Print formatted report
           - Include support values
        """
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
