# CSCI 544 - Applied Natural Language Processing

## Assignment 1

- Implemented sentiment analysis on Amazon reviews dataset
- Data cleaning
- Preprocesing
  - Removing stop words
  - Lemmatization
- TF-IDF feature extraction
- Train Model
  - Perceptron
  - SVM
  - Logistic Regression
  - Naive Bayes

## Assignment 2

- Implemented sentiment analsys on Amazon reviews dataset using Word2Vec embeddings and Neural networks
- Data cleaning
- Preprocessing
  - Removing Stopwords
  - Lemmatization
- Word2Vec Feature Extraction
  - Using Google Word2Vec embeddings
  - Using Custom Word2Vec embeddings from dataset
- Train Model (16 models)
  | Sl No | Model | Classification Type | Word2Vec Input Type | Word2Vec Model |
  |-------|------------|---------------------|---------------------|----------------|
  | 1 | Perceptron | Binary | Mean | Pretrained |
  | 2 | Perceptron | Binary | Mean | Custom |
  | 3 | SVM | Binary | Mean | Pretrained |
  | 4 | SVM | Binary | Mean | Custom |
  | 5 | FFNN | Binary | Mean | Pretrained |
  | 6 | FFNN | Ternary | Mean | Pretrained |
  | 7 | FFNN | Binary | Concat | Pretrained |
  | 8 | FFNN | Ternary | Concat | Pretrained |
  | 9 | FFNN | Binary | Mean | Custom |
  | 10 | FFNN | Ternary | Mean | Custom |
  | 11 | FFNN | Binary | Concat | Custom |
  | 12 | FFNN | Ternary | Concat | Custom |
  | 13 | CNN | Binary | 50x300 | Pretrained |
  | 14 | CNN | Binary | 50x300 | Custom |
  | 15 | CNN | Ternary | 50x300 | Pretrained |
  | 16 | CNN | Ternary | 50x300 | Custom |

## Assignment 3

- Implement Parts of Speech tagging 
- HMM model
- Greddy Decoding 

