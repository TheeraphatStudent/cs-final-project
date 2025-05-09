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
    def __init__(self, model_name):
        """
        Initialize the ModelPerformance tracker
        
        Args:
            model_name (str): Name of the model for tracking
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
        
    def update_history(self, metrics):
        """
        Update training history with new metrics
        
        Args:
            metrics (dict): Dictionary containing training metrics
        """
        for key in self.history.keys():
            if key in metrics:
                self.history[key].append(metrics[key])
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate various performance metrics
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array, optional): Predicted probabilities
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate ROC curve if probabilities are provided
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def plot_training_history(self, save_path=None):
        """Plot training history metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_accuracy'], label='Train')
        plt.plot(self.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot precision and recall
        plt.subplot(2, 2, 3)
        plt.plot(self.history['val_precision'], label='Precision')
        plt.plot(self.history['val_recall'], label='Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        # Plot F1 score
        plt.subplot(2, 2, 4)
        plt.plot(self.history['val_f1'], label='F1 Score')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """Plot ROC curve"""
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
    
    def save_metrics(self, save_path):
        """Save metrics to a JSON file"""
        metrics_data = {
            'model_name': self.model_name,
            'history': self.history,
            'metrics_history': self.metrics_history
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
    
    def load_metrics(self, load_path):
        """Load metrics from a JSON file"""
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                metrics_data = json.load(f)
                self.history = metrics_data['history']
                self.metrics_history = metrics_data['metrics_history']
    
    def get_latest_metrics(self):
        """Get the latest performance metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
