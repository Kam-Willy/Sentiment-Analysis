---

# Sentiment Analysis on Amazon Food Reviews

![Sentiment Analysis](https://img.shields.io/badge/Sentiment-Analysis-blue.svg)

This repository contains code for a sentiment analysis project on Amazon food reviews, using various sentiment analysis methods like **NLTK VADER**, **BERT pretrained model**, and custom neural networks in **TensorFlow** and **PyTorch**. The project aims to classify review sentiments (positive, neutral, negative) and compare performance across multiple models.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Modeling Techniques](#modeling-techniques)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview
Sentiment analysis is an NLP task to determine the emotional tone behind text data. This project analyzes Amazon fine food reviews and uses various models to classify reviews into positive, neutral, or negative sentiments. 
The project includes exploratory data analysis (EDA) and model evaluation and comparison.

## Dataset
The Amazon Fine Food Reviews dataset includes reviews with fields such as `ReviewText` and `Score`. The `Score` field (ranging from 1 to 5) is converted into sentiment labels as follows:
- **1-2**: Negative
- **3**: Neutral
- **4-5**: Positive

## Modeling Techniques
The repository contains implementations of the following models:
1. **NLTK VADER**: A rule-based model to analyze sentiments quickly.
2. **BERT Pretrained Model**: A transformer-based model from Hugging Face for advanced NLP tasks.
3. **TensorFlow Neural Network**: A custom neural network with TensorBoard monitoring.
4. **PyTorch Neural Network**: A custom neural network model using PyTorch with DataLoader and batching.

Each model is evaluated on accuracy and F1-score to determine the best-performing model for sentiment analysis on the dataset.

## Installation

### Prerequisites
- Python 3.8+

Install the necessary packages:
1. nltk
2. torch
3. numpy
4. pandas
5. seaborn
6. tqdm
7. tensorflow
8. matplotlib
9. transformers: BertTokenizer, BertForSequenceClassification
10. sklearn


### Requirements
The project relies on several libraries for data processing, model building, and evaluation:
- **Pandas**: For data manipulation.
- **NLTK**: For VADER sentiment analysis.
- **Transformers**: For using the BERT model.
- **TensorFlow** and **PyTorch**: For deep learning models.
- **TQDM**: For displaying progress bars during processing.

## Usage

### 1. Exploratory Data Analysis and Preprocessing
Run `eda_preprocessing.py` to analyze and preprocess the dataset. This script performs:
- Text cleaning and tokenization.
- Sentiment label creation based on score.
- Exploratory data analysis with plots.

### 2. Running the Models
Each model has its dedicated script:
- **NLTK VADER Model**: Run `vader_model.py` to analyze sentiments with VADER.
- **BERT Model**: Run `bert_model.py` for BERT-based sentiment analysis.
- **TensorFlow Model**: Run `tensorflow_model.py` to train and evaluate the TensorFlow neural network.
- **PyTorch Model**: Run `pytorch_model.py` to train and evaluate the PyTorch neural network.

### 3. Evaluation and Comparison
Run `evaluate_models.py` to compare models and display the classification report and plots of performance metrics.

### Example Code
Here’s how to run the VADER model on a batch of text data:

```python
from vader_model import VaderSentimentAnalyzer

# Initialize VADER analyzer
vader = VaderSentimentAnalyzer()

# Analyze a sample text
sample_text = "This product was excellent! I highly recommend it."
print("Sentiment:", vader.analyze(sample_text))
```

## Results
The project outputs include:
- Classification reports for each model (accuracy, precision, recall, and F1-score).
- Plots comparing model performances.

### Example Output
| Model       | Accuracy | F1 Score |
|-------------|----------|----------|
| NLTK VADER  | 78.5%    | 0.75     |
| BERT        | 85.3%    | 0.82     |
| TensorFlow  | 84.0%    | 0.80     |
| PyTorch     | 83.7%    | 0.79     |

TensorBoard can be accessed for detailed TensorFlow model training information.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

---
