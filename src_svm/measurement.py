import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import json
import os

class ModelPerformanceSVM:
    def __init__(self, model_name):
        """
        Initialize the enhanced ModelPerformance tracker
        
        Args:
            model_name (str): Name of the model
        """
        self.model_name = model_name
        self.history = {
            'train_scores': [],
            'val_scores': [],
            'cv_scores': [],
            'feature_importance': {},
            'hyperparameters': {}
        }
        self.metrics_history = []
        self.model_comparison = {}
        
    def update_history(self, history_data):
        """
        Update training history with new metrics
        
        Args:
            history_data (dict): Dictionary containing training metrics
        """
        for key in self.history:
            if key in history_data:
                if isinstance(history_data[key], list):
                    self.history[key].extend(history_data[key])
                else:
                    self.history[key] = history_data[key]
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            metrics.update({
                'roc_auc': float(roc_auc),
                'average_precision': float(avg_precision),
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            })
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'balanced_accuracy': float((metrics['precision'] + metrics['recall']) / 2)
        })
        
        self.metrics_history.append(metrics)
        return metrics
    
    def plot_comprehensive_performance(self, save_path=None):
        if not self.metrics_history:
            print("No metrics available for plotting")
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name} - Comprehensive Performance Analysis', fontsize=16)
        
        # 1. Confusion Matrix
        cm = np.array(latest_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve (if available)
        if 'roc_curve' in latest_metrics:
            fpr = latest_metrics['roc_curve']['fpr']
            tpr = latest_metrics['roc_curve']['tpr']
            roc_auc = latest_metrics['roc_auc']
            
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend(loc="lower right")
        
        # 3. Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            latest_metrics['accuracy'],
            latest_metrics['precision'],
            latest_metrics['recall'],
            latest_metrics['f1']
        ]
        
        bars = axes[0, 2].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0, 2].set_title('Performance Metrics')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_ylim(0, 1)
        
        for bar, value in zip(bars, metric_values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        if self.history['cv_scores']:
            axes[1, 0].boxplot(self.history['cv_scores'])
            axes[1, 0].set_title('Cross-Validation Scores')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_xticklabels(['CV Scores'])
        
        if self.history['train_scores'] and self.history['val_scores']:
            train_sizes = range(1, len(self.history['train_scores']) + 1)
            axes[1, 1].plot(train_sizes, self.history['train_scores'], label='Training Score')
            axes[1, 1].plot(train_sizes, self.history['val_scores'], label='Validation Score')
            axes[1, 1].set_title('Learning Curve')
            axes[1, 1].set_xlabel('Training Size')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
        
        if self.history['feature_importance']:
            importance = self.history['feature_importance']
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
            
            feature_names = list(top_features.keys())
            importance_values = list(top_features.values())
            
            axes[1, 2].barh(range(len(feature_names)), importance_values)
            axes[1, 2].set_yticks(range(len(feature_names)))
            axes[1, 2].set_yticklabels(feature_names)
            axes[1, 2].set_xlabel('Importance')
            axes[1, 2].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Enhanced confusion matrix visualization
        
        Args:
            cm (array): Confusion matrix
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(8, 6))
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        cmap = sns.color_palette(colors, as_cmap=True)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                   xticklabels=['Safe', 'Malicious'],
                   yticklabels=['Safe', 'Malicious'])
        
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """
        Enhanced ROC curve visualization
        
        Args:
            fpr (array): False positive rates
            tpr (array): True positive rates
            roc_auc (float): ROC AUC score
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        # Add grid and styling
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{self.model_name} - Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        
        # Add AUC score text
        plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot precision-recall curve
        
        Args:
            y_true (array): True labels
            y_pred_proba (array): Prediction probabilities
            save_path (str): Path to save plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=3, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'{self.model_name} - Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=12)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance_dict, save_path=None, top_n=15):
        """
        Enhanced feature importance visualization
        
        Args:
            importance_dict (dict): Feature importance dictionary
            save_path (str): Path to save plot
            top_n (int): Number of top features to display
        """
        if not importance_dict:
            print("No feature importance data available")
            return
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        feature_names = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        # Create horizontal bar plot
        plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))
        
        # Create color gradient based on importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_values)))
        
        bars = plt.barh(range(len(feature_names)), importance_values, color=colors)
        
        # Customize plot
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'{self.model_name} - Top {top_n} Feature Importance', fontsize=14)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, importance_values)):
            plt.text(bar.get_width() + max(importance_values) * 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curve(self, train_sizes, train_scores, val_scores, save_path=None):
        """
        Plot learning curve
        
        Args:
            train_sizes (array): Training set sizes
            train_scores (array): Training scores
            val_scores (array): Validation scores
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Examples', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'{self.model_name} - Learning Curve', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics(self, filepath):
        """
        Save comprehensive metrics to JSON file
        
        Args:
            filepath (str): Path to save metrics
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'model_name': self.model_name,
            'history': self.history,
            'metrics_history': self.metrics_history,
            'model_comparison': self.model_comparison,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
    
    def load_metrics(self, filepath):
        """
        Load metrics from JSON file
        
        Args:
            filepath (str): Path to metrics file
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.history = data.get('history', self.history)
            self.metrics_history = data.get('metrics_history', [])
            self.model_comparison = data.get('model_comparison', {})
            print(f"Metrics loaded from {filepath}")
        else:
            print(f"Metrics file {filepath} not found")
    
    def get_latest_metrics(self):
        """
        Get the latest performance metrics
        
        Returns:
            dict: Latest metrics or None if no metrics available
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def print_detailed_report(self, y_true, y_pred, y_pred_proba=None):
        """
        Print detailed classification report
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array): Prediction probabilities
        """
        print(f"\n{'='*60}")
        print(f"{self.model_name} - Detailed Performance Report")
        print(f"{'='*60}")
        
        # Basic metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        
        print(f"\nBasic Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Avg Precision: {metrics['average_precision']:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        report = classification_report(y_true, y_pred, target_names=['Safe', 'Malicious'])
        print(report)
        
        print(f"{'='*60}")
    
    def compare_models(self, other_metrics, other_name):
        """
        Compare current model with another model
        
        Args:
            other_metrics (dict): Metrics from another model
            other_name (str): Name of the other model
        """
        current_metrics = self.get_latest_metrics()
        if not current_metrics:
            print("No current metrics available for comparison")
            return
        
        comparison = {
            'current_model': current_metrics,
            'other_model': other_metrics,
            'differences': {}
        }
        
        # Calculate differences
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in current_metrics and metric in other_metrics:
                diff = current_metrics[metric] - other_metrics[metric]
                comparison['differences'][metric] = diff
        
        self.model_comparison[other_name] = comparison
        
        # Print comparison
        print(f"\nModel Comparison: {self.model_name} vs {other_name}")
        print("-" * 50)
        for metric, diff in comparison['differences'].items():
            print(f"{metric.capitalize()}: {diff:+.4f}")
        
        return comparison 