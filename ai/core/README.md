## Ai Model Core

Ai model running with python11 on container

**To running model with local machine**

Before running must be install followed by [requirement.txt](ai/core/model/code/requirements.txt)

Go to [code](/ai/core/model/code) and run `python main.py`

**To running model with container**

Before runing please make sure cuda can be usaged by testing with `python test_cuda.py`

if running with `container` testing with 
```bash
podman run --device nvidia.com/gpu=all
``` 

```bash
podman compose -f ai/core/model/code/cuda.docker-compose.yaml up -d
```

After it working, followed this step

```bash
podman compose -f ai/core/model/code/docker-compose.yaml up -d -build --remove-orphans
```

### model Usage

- random forest classifier
- svm
- xgboost

## Reference

### Model

- [What is the difference between SVC and SVM in scikit learn](https://stackoverflow.com/questions/27912872/what-is-the-difference-between-svc-and-svm-in-scikit-learn)

- [GPU-accelerated SVM alternative](https://thundersvm.readthedocs.io/en/latest/)

- [A New GPU Implementation of Support Vector Machines for Fast Hyperspectral Image Classification
](https://www.mdpi.com/2072-4292/12/8/1257)

- [RF Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XG boots](https://xgboost.readthedocs.io/en/stable/)

### Tools

- [Unified engine for large-scale data analytics](https://spark.apache.org/)

### Project

- [ShotDroid](https://github.com/kp300/shotdroid)
ShotDroid is a pentesting tool for android. There are 3 tools that have their respective functions:

- [Malicious URLs Detection using Machine Learning](https://github.com/Youssef-Daouayry/Malicious-URLs-Detection-using-Machine-Learning)

### Research

- [From Past to Present: A Survey of Malicious URL Detection
Techniques, Datasets and Code Repositories](https://arxiv.org/pdf/2504.16449)
- [Machine Learning for Malicious URL Detection](https://www.researchgate.net/publication/347620249_Machine_Learning_for_Malicious_URL_Detection)
- [Machine Learning-Powered Malicious Website Detection System](https://www.researchgate.net/publication/388429921_Machine_Learning-Powered_Malicious_Website_Detection_System)
- [Paper - Mjekokcus QVE Bataotkcl QskliMjodkla Eajrlkli](https://www.scribd.com/presentation/519481206/Seminar-PPT-Final)
- [Extraction feature vectors from url string from malicious url](https://medium.com/data-science/extracting-feature-vectors-from-url-strings-for-malicious-url-detection-cbafc24737a)
- [What is a URL](https://developer.mozilla.org/en-US/docs/Learn_web_development/Howto/Web_mechanics/What_is_a_URL)
- [What is a Domain Name?](https://developer.mozilla.org/en-US/docs/Learn_web_development/Howto/Web_mechanics/What_is_a_domain_name)
- [What is a top-level domain (TLD)?](https://www.cloudflare.com/learning/dns/top-level-domain/)
- [What Top-level Domains attract Phishers?](https://www.cybercrimeinfocenter.org/what-top-level-domains-attract-phishers)
- [Saving trained model in Python](https://neptune.ai/blog/saving-trained-model-in-python)

### Deployment
- [Ml flow](https://mlflow.org/)
- [Ml flow docker ](https://mlflow.org/docs/latest/ml/docker/)
- [Ml flow sklearn integration](https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/index.html)
- [Ml flow with exist model](https://mlflow.org/docs/latest/ml/model-registry/)
- [Databrick](https://docs.databricks.com/aws/en/mlflow/)
- [Github container registry](https://github.com/docker/login-action#github-container-registry)