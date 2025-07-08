## Algorithm: `Neural Network for Binary Classification`

- Algorithm Type: Feed-forward Neural Network (Multi-layer Perceptron)
- Architecture: 4-layer neural network with dropout regularization
- Classification Type: Binary classification (Safe / Malicious URLs)

## How it work

```
Input Layer (12 features) 
    ↓
Dense Layer 1: 64 neurons + ReLU activation + Dropout(0.3)
    ↓
Dense Layer 2: 32 neurons + ReLU activation + Dropout(0.2)
    ↓
Dense Layer 3: 16 neurons + ReLU activation
    ↓
Output Layer: 1 neuron + Sigmoid activation
```

## URL Analysis Algorithms

- File extension analysis (.exe, .zip, .pdf, etc.)
- Keyword matching (login, signin, bank, etc.)
- Domain spoofing detection
- IP address validation

##  Feature Calculation

- URL length metrics
- Character counting (dots, slashes, digits, special chars)
- Path and query length analysis
- Domain structure analysis