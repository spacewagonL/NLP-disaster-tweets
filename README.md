# Disaster Tweet Classification using Recurrent Neural Networks (RNN)

## Project Overview
This project focuses on classifying tweets as **disaster-related (1)** or **not disaster-related (0)** using **Natural Language Processing (NLP)** and **deep learning**. The dataset, provided by Kaggle, consists of approximately **7,600 tweets**, which were preprocessed and used to train a **Recurrent Neural Network (RNN) with Bidirectional LSTM**.

## Dataset
- **Source**: Kaggle - NLP with Disaster Tweets ([competition link](https://www.kaggle.com/competitions/nlp-getting-started))
- **Size**: ~7,600 tweets
- **Labels**:
  - `1`: Disaster-related tweet
  - `0`: Non-disaster tweet
- **Structure**:
  - `train.csv`: Contains labeled tweets for training
  - `test.csv`: Contains unlabeled tweets for prediction
  - `sample_submission.csv`: Submission format for Kaggle

## Approach
### 1. Preprocessing
- Tokenization and text cleaning (removing URLs, special characters, and stopwords)
- Sequence conversion using Keras `Tokenizer`
- Padding sequences to ensure uniform input length

### 2. Model Architecture
- **Embedding Layer**: Converts words into dense vector representations
- **Bidirectional LSTM**: Captures both forward and backward dependencies in text
- **Dropout & L2 Regularization**: Reduces overfitting
- **Dense Layers**: Fully connected layers for classification

### 3. Training Optimization
- **Batch size**: 32
- **Optimizer**: Adam
- **Loss function**: Binary Crossentropy
- **Early Stopping**: Prevents overfitting by stopping training when validation loss starts increasing
- **Class Weights**: Adjusted to handle imbalanced classes

## Results
- **Validation F1 Score**: **0.7197**
- **Validation Accuracy**: **0.7756**
- **Predicted Probability Range**: **0.0003 to 0.9996**
- **Overfitting Mitigation**: Regularization techniques reduced overfitting, but further improvements in validation loss are needed.

## Challenges & Future Improvements
- **Overfitting**: While regularization improved generalization, validation loss can still be reduced.
- **Threshold Adjustment**: Further tuning of the classification threshold may improve F1 score.
- **Data Augmentation**: Expanding the dataset through augmentation techniques could enhance model robustness.
- **Alternative Models**: Exploring Transformer models (e.g., BERT, RoBERTa) and TPUs may improve classification performance.


