# วิธีการทำงานของ Model สำหรับ URL Classification System

## 1. Feature
### 1.1 URL Length Metrics
- Total URL Length: $L_{total} = \sum_{i=1}^{n} |c_i|$ where $c_i$ is each character
- Path Length: $L_{path} = \sum_{i=1}^{m} |p_i|$ where $p_i$ is each path component
- Query Length: $L_{query} = \sum_{i=1}^{k} |q_i|$ where $q_i$ is each query parameter

**Sample:**
URL: `https://example.com/path/to/page?param1=value1&param2=value2`
- $L_{total} = 52$ characters
- $L_{path} = 12$ characters (`/path/to/page`)
- $L_{query} = 24$ characters (`param1=value1&param2=value2`)

### 1.2 Character Analysis
- Special Character: $S = \sum_{i=1}^{n} \mathbb{1}(c_i \in \Omega)$ where $\Omega$ is the set of special characters
- Digit Count: $D = \sum_{i=1}^{n} \mathbb{1}(c_i \in \{0,1,...,9\})$
- Dot Count: $P = \sum_{i=1}^{n} \mathbb{1}(c_i = '.')$

**Sample:**
URL: `https://example123.com/path?param=value`
- $S = 8$ (counting `://./?=`)
- $D = 3$ (counting `123`)
- $P = 1$ (counting `.`)

## 2. Neural Network Architecture

### 2.1 Layer Structure
- Input Layer: $X \in \mathbb{R}^{d}$ where $d$ is the number of features
- Hidden Layer 1: $H_1 = \max(0, W_1X + b_1)$ where $W_1 \in \mathbb{R}^{64 \times d}$
- Dropout Layer 1: $D_1 = \text{Dropout}(H_1, p=0.3)$
- Hidden Layer 2: $H_2 = \max(0, W_2D_1 + b_2)$ where $W_2 \in \mathbb{R}^{32 \times 64}$
- Dropout Layer 2: $D_2 = \text{Dropout}(H_2, p=0.2)$
- Hidden Layer 3: $H_3 = \max(0, W_3D_2 + b_3)$ where $W_3 \in \mathbb{R}^{16 \times 32}$
- Output Layer: $Y = \sigma(W_4H_3 + b_4)$ where $W_4 \in \mathbb{R}^{1 \times 16}$

### 2.2 Activation Functions
- ReLU: $f(x) = \max(0, x)$
- Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$

## 3. Model Training

### 3.1 Loss Function
Binary Cross-Entropy Loss:
$$L(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Sample:**
$N=2$:
- $y = [1, 0]$
- Predicted: $\hat{y} = [0.8, 0.2]$
- $L(y, \hat{y}) = -\frac{1}{2}[1 \cdot \log(0.8) + 0 \cdot \log(0.2) + 0 \cdot \log(0.2) + 1 \cdot \log(0.8)] \approx 0.223$

### 3.2 Optimization
Adam Optimizer:
- $m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$
- $v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$
- Bias correction: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
- Parameter update: $\theta_t = \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

## 4. Performance Metrics

### 4.1 Classification Metrics
- Accuracy: $A = \frac{TP + TN}{TP + TN + FP + FN}$
- Precision: $P = \frac{TP}{TP + FP}$
- Recall: $R = \frac{TP}{TP + FN}$
- F1 Score: $F1 = 2\frac{P \times R}{P + R}$

**Sample:**
Test set of 100 URLs:
- True Positives (TP) = 45
- True Negatives (TN) = 40
- False Positives (FP) = 10
- False Negatives (FN) = 5

Then:
- $A = \frac{45 + 40}{100} = 0.85$ (85% accuracy)
- $P = \frac{45}{45 + 10} = 0.818$ (81.8% precision)
- $R = \frac{45}{45 + 5} = 0.9$ (90% recall)
- $F1 = 2\frac{0.818 \times 0.9}{0.818 + 0.9} \approx 0.857$ (85.7% F1 score)

### 4.2 Confusion Matrix
$$\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}$$

## 5. Feature Scaling

### 5.1 StandardScaler
$$x_{scaled} = \frac{x - \mu}{\sigma}$$
- $\mu$ = feature
- $\sigma$ = standard deviation

**Sample:**
URL lengths in a dataset:
- Origin: [10, 15, 20, 25, 30]
- $\mu = 20$
- $\sigma = 7.07$
- Scaled values: [-1.41, -0.71, 0, 0.71, 1.41]

## 6. Prediction Confidence

### 6.1 Confidence Score
URL = $u$:
$$C(u) = |P(y=1|u) - 0.5| \times 2$$
$P(y=1|u)$ model's predicted = maliciousness

$P(y=0|u)$ model's predicted = safe

**Sample:**
3 URLs:
1. $P(y=1|u_1) = 0.9$ → $C(u_1) = |0.9 - 0.5| \times 2 = 0.8$ (high confidence)
2. $P(y=1|u_2) = 0.6$ → $C(u_2) = |0.6 - 0.5| \times 2 = 0.2$ (low confidence)
3. $P(y=1|u_3) = 0.5$ → $C(u_3) = |0.5 - 0.5| \times 2 = 0$ (no confidence)

## 7. Feature Importance

### 7.1 Contribution
feature = $f$:
$$I(f) = \sum_{i=1}^{n} |w_i|$$
$w_i$ is weights -> feature $f$ to the first hidden layer