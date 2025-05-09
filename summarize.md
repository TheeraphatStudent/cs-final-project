# URL Classification System Summary

## Overview
This is a machine learning-based system for detecting malicious URLs. The system uses a neural network model trained on various URL features to classify URLs as either safe or malicious.

## System Architecture

### 1. Data Processing (`DataProcessor` class)
- **Input**: Excel file containing URL data
- **Output**: Processed CSV file with extracted features
- **Key Components**:
  - URL component extraction (subdomain, domain, suffix, path, query, fragment)
  - URL metrics calculation (length, special characters, etc.)
  - Suspicious pattern detection

### 2. Model Architecture (`URLClassifier` class)
- **Neural Network Structure**:
  ```
  Input Layer → Dense(64, ReLU) → Dropout(0.3) → 
  Dense(32, ReLU) → Dropout(0.2) → 
  Dense(16, ReLU) → 
  Output Layer(1, Sigmoid)
  ```
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy

### 3. Performance Measurement (`ModelPerformance` class)
- Tracks and visualizes model performance metrics
- Generates plots for training history and confusion matrix
- Saves metrics in JSON format

## Mathematical Concepts

### 1. Feature Engineering
- **URL Length**: L = length(url)
- **Special Characters Count**: S = count(special_chars)
- **Path Length**: P = length(path)
- **Query Length**: Q = length(query)

### 2. Model Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)

### 3. Neural Network
- **Activation Functions**:
  - ReLU: f(x) = max(0, x)
  - Sigmoid: f(x) = 1 / (1 + e^(-x))
- **Dropout**: Random deactivation of neurons (p = 0.3, 0.2)
- **Binary Cross-Entropy Loss**: 
  L = -[y * log(p) + (1-y) * log(1-p)]

## Algorithms

### 1. URL Processing
```python
def extract_url_components(url):
    1. Remove protocol (http/https)
    2. Split into domain and path
    3. Extract subdomain, domain, suffix
    4. Parse path, query, fragment
```

### 2. Feature Extraction
```python
def extract_features(df):
    1. Select numerical features
    2. Scale features using StandardScaler
    3. Return feature matrix
```

### 3. Model Training
```python
def train(dataset_path):
    1. Load and preprocess data
    2. Split into train/test sets (80/20)
    3. Create and compile model
    4. Train with early stopping
    5. Calculate and save metrics
```

### 4. Prediction Pipeline
```python
def predict(url):
    1. Check if known safe domain
    2. Extract URL components
    3. Calculate features
    4. Scale features
    5. Make prediction
    6. Generate explanation
```

## Data Storage

### 1. Model Files
- `xxx.keras`: Trained neural network model
- `xxx.joblib`: Feature scaler for preprocessing

### 2. Metrics and Visualizations
- `metrics/xxx.json`: Performance metrics
- `plots/xxx.png`: Training progress
- `plots/xxx.png`: Model and Classification structure

## Key Features

1. **URL Analysis**:
   - Protocol detection (HTTP/HTTPS)
   - Domain structure analysis
   - Path and query parameter analysis
   - Special character detection

2. **Security Checks**:
   - Suspicious extension detection
   - Known malicious pattern matching
   - Legitimate domain verification
   - IP address detection

3. **Performance Monitoring**:
   - Real-time accuracy tracking
   - Confusion matrix visualization
   - ROC curve analysis
   - Training history plots

4. **Model Management**:
   - Automatic model saving/loading
   - Feature scaling persistence
   - Performance metrics storage
   - Model architecture visualization 