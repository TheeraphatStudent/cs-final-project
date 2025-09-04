# Malicious URL Detection using Machine Learning

To train an ai model to predict an url is safe or not

**If wanna change an feature of ai, go to section of feature etraxtion and implement it kub!**

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

# Setup environment

1. Create **.venv**
```
python -m venv .venv
```

2. Active .venv
```
source .\.venv\bin\activate
```

**if permission issue running chmod +x to `activate`**

3. Install requirement
```
.\.venv\bin\pip install -r requirements.txt
```

# Problem with jupyter notebook

- Select kernel from venv
- If can't try to close and open ide again it will be working kub :)