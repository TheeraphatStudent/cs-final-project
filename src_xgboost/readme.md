# XGBoost URL Malicious Detection System

## Algorithm: `XGBoost for Binary Classification`

- Algorithm Type: Gradient Boosting Decision Trees
- Architecture: Ensemble of decision trees with boosting
- Classification Type: Binary classification (Safe / Malicious URLs)

## How it work

```
Input Features (12 features)
    ↓
XGBoost Classifier
- n_estimators: 100 trees
- max_depth: 6 levels
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
    ↓
Output: Probability score (0-1)
```

## Performance Results

### Model Performance Metrics:
- **Accuracy**: 93.31%
- **Precision**: 94.41%
- **Recall**: 98.30%
- **F1 Score**: 96.32%

### Confusion Matrix:
- **True Negatives**: 1,378 (correctly identified safe URLs)
- **False Positives**: 1,205 (safe URLs flagged as malicious)
- **False Negatives**: 353 (malicious URLs missed)
- **True Positives**: 20,361 (correctly identified malicious URLs)

## Feature Importance Analysis

### Top 3 Most Important Features:
1. **f0 (url_length)**: 908.0 - URL length is the most critical feature
2. **f5 (num_digits)**: 877.0 - Number of digits is very important
3. **f1 (path_length)**: 838.0 - Path length is highly significant

### Feature Mapping:
| Feature | Name | Description |
|---------|------|-------------|
| f0 | url_length | Total length of the URL |
| f1 | path_length | Length of the URL path |
| f2 | query_length | Length of the query parameters |
| f3 | num_dots | Number of dots in the URL |
| f4 | num_slashes | Number of forward slashes |
| f5 | num_digits | Number of digits in the URL |
| f6 | num_special_chars | Number of special characters |
| f7 | has_suspicious_extension | Binary flag for suspicious file extensions |
| f8 | has_suspicious_keyword | Binary flag for suspicious keywords |
| f9 | is_legitimate_domain | Binary flag for legitimate domains |
| f10 | has_https | Binary flag for HTTPS protocol |
| f11 | has_http | Binary flag for HTTP protocol |

## URL Analysis Algorithms

- File extension analysis (.exe, .zip, .pdf, etc.)
- Keyword matching (login, signin, bank, etc.)
- Domain spoofing detection
- IP address validation

## Feature Calculation

- URL length metrics
- Character counting (dots, slashes, digits, special chars)
- Path and query length analysis
- Domain structure analysis

## Key Insights

- **URL length** is the most important feature for detecting malicious URLs
- **Number of digits** in URLs is a strong indicator of maliciousness
- **Path length** is also very important
- **Protocol type** (HTTP/HTTPS) has relatively low importance
- The model relies heavily on **structural features** rather than content-based features
- Malicious URLs tend to be longer, contain more digits, and have longer paths

## Usage

1. Install requirements: `pip install -r requirements.txt`
2. Run the model: `python __main__.py`
3. Choose options:
   - Check URL: Test individual URLs
   - Train model: Retrain with new data
   - Model performance: View metrics
   - Visualize model: Generate feature importance plot