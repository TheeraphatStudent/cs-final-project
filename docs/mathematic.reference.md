# เอาไว้ทำวิจัย

## 1. URL Analysis

### URL Length Metrics
- Formula: $L_{total} = \sum_{i=1}^{n} |c_i|$
- Link: https://www.rfc-editor.org/rfc/rfc3986

### Character Analysis
- Formula: $S = \sum_{i=1}^{n} \mathbb{1}(c_i \in \Omega)$
- Link: https://arxiv.org/abs/1509.01626

## 2. Neural Network Architecture

### Layer Structure
- Formulas: $H_1 = \max(0, W_1X + b_1)$, $Y = \sigma(W_4H_3 + b_4)$
- Link: https://www.deeplearningbook.org/

### Activation Functions
- ReLU: $f(x) = \max(0, x)$
- Link: https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf

- Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Link: http://neuralnetworksanddeeplearning.com/

## 3. Model Training

### Binary Cross-Entropy Loss
- Formula: $L(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
- Link: https://www.microsoft.com/en-us/research/people/cmbishop/

### Adam Optimizer
- Formulas: $m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$
- Link: https://arxiv.org/abs/1412.6980

### Other

- [Classification Metrics](https://mitpress.mit.edu/books/introduction-machine-learning)
- [F1 Score](https://nlp.stanford.edu/IR-book/)
- [StandardScaler](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Confidence Score](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)

## Etc...
   - [A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/)
   - [URL Classification Research](https://ieeexplore.ieee.org/document/8809871)
   - [Neural Network Implementation](https://www.manning.com/books/deep-learning-with-python-second-edition)
   - [Model Evaluation](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
