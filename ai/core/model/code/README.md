# Malicious URL Detection using Machine Learning

## Ai model

- [How to make an ai model](https://www.netguru.com/blog/how-to-make-an-ai-model)
- [Everything you need to know about AI model training](https://www.labellerr.com/blog/everything-you-need-to-know-about-ai-model-training/) 
- [Future-proof your data preparation strategy for machine learning](https://www.kellton.com/kellton-tech-blog/future-proof-your-data-preparation-strategy-for-machine-learning)
- [Model Tuning](https://www.ibm.com/think/topics/model-tuning)
- [Stratefies for fins tuning LLM](https://www.capellasolutions.com/blog/strategies-for-fine-tuning-large-language-models)

### Output Architecher
- [GRU vs Bi-GRU: Which one is going to win?](https://vtiya.medium.com/gru-vs-bi-gru-which-one-is-going-to-win-58a45ede5fba)
- [Bidirectional RNN, Bidirectional LSTM, Bidirectional GRU](https://medium.com/@abhishekjainindore24/bidirectional-rnn-bidirectional-lstm-bidirectional-gru-dba4476c98bc)
- 

## Dataset

- [Dataset Schema](https://www.opendatasoft.com/en/glossary/dataset-schema/)
- [How do words become numbers in AI models?](https://medium.com/@rgalvg/how-do-words-become-numbers-in-ai-models-3d399dbcbb79)
- [Text to numeric representation in NLP: A beginner-friendly guide](https://medium.com/@ketan.patel_46870/text-to-numeric-representation-in-nlp-a-beginner-friendly-guide-9e68c8f8d07c)

## Python Library

*   **tensorflow:** The go-to library for building and training complex machine learning and deep learning models, especially neural networks.
*   **xgboost:** A powerful and highly efficient library for creating gradient boosting models, often used to win machine learning competitions.
*   **numpy:** The fundamental package for numerical computing, providing powerful array objects and mathematical functions.
*   **scikit-learn:** A comprehensive and user-friendly library for a wide range of machine learning tasks like classification, regression, and clustering.
*   **pandas:** An essential tool for data manipulation and analysis, offering flexible data structures like the DataFrame.
*   **nltk:** A complete toolkit for natural language processing (NLP), used for working with and analyzing human language text.
*   **openpyxl:** A specialized library for reading from and writing to Microsoft Excel files (`.xlsx` format).
*   **matplotlib:** The foundational library for creating a wide variety of static, animated, and interactive plots and visualizations.
*   **seaborn:** Built on top of matplotlib, this library makes it easier to create more attractive and statistically informative graphics.
*   **requests:** The standard library for making simple and user-friendly HTTP requests to interact with web services and APIs.
*   **beautifulsoup4:** A library perfect for web scraping, allowing you to pull data out of HTML and XML files.
*   **urllib3:** A powerful, low-level HTTP client that provides fine-grained control over connections and requests, often used by other libraries like `requests`.
*   **joblib:** A set of tools to provide lightweight pipelining in Python, particularly useful for saving and loading machine learning models and running parallel computations.
*   **IPython:** An enhanced interactive Python shell that offers a more powerful and user-friendly command-line experience than the standard Python interpreter.
*   **scipy:** A core scientific computing library that builds on NumPy to provide a large collection of algorithms for optimization, linear algebra, and signal processing.
*   **wordcloud:** A library for generating word clouds, which are visual representations of text data where the size of each word is proportional to its frequency in the text.

# Install tools
**This project was develop with linux base Fedora42**

### Install build tools

```bash
sudo yum install cmake make gcc-c++
```

### Nvidia driver container

**Install nvcc**

`Must be direct install nvidia driver first naja!`

- [Full guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

```bash
sudo dnf config-manager addrepo --from-repofile https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
sudo dnf clean all
sudo dnf -y install cuda-toolkit-13-0
```

```bash
sudo dnf install nvidia-open --allowerasing
```

```bash
sudo dnf -y install cuda-drivers --allowerasing
```

```bash
sudo dnf install kmod-nvidia-open-dkms --allowerasing
```

**Install nvidia container toolkit**
```bash
sudo dnf copr enable @ai-ml/nvidia-container-toolkit
```

**Install nvidia driver**
```bash
sudo dnf install xorg-x11-drv-nvidia-cuda
sudo dnf install nvidia-container-toolkit nvidia-container-toolkit-selinux
```

**Generate the Container Device Interface (CDI) configuration file**
```bash
sudo nvidia-ctk cdi generate -output /etc/cdi/nvidia.yaml
```

**Testing**
```bash
podman run --device nvidia.com/gpu=all --rm fedora:latest nvidia-smi
```

Output
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   55C    P0             15W /   75W |      15MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

----

Test with compose


Container
```yaml
services:
  test:
    image: nvidia/cuda:12.9.0-base-ubuntu22.04
    command: nvidia-smi
    devices:
      - nvidia.com/gpu=all
```

Running command
```bash
podman compose -f ./cuda.docker-compose.yaml up -d
```

## Additional Setup

1. Find cuda core install location

```
/usr/local/
```