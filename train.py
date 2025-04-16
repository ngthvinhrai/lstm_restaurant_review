# import sys
# sys.path.append('../')

from NeuralNetworks.cupy_NeuralNetworks.Model import Sequential
from NeuralNetworks.cupy_NeuralNetworks.Layer import Dense, LSTM, RNN, ModernLSTM, Embedding
from NeuralNetworks.cupy_NeuralNetworks.Activation import Sigmoid, Tanh, Softmax, Linear
from NeuralNetworks.cupy_NeuralNetworks.Loss import CrossEntropy
from NeuralNetworks.cupy_NeuralNetworks.Optimizer import GradientDescent, Momentum
from .data_preprocessing import DataPreprocessing
from NeuralNetworks.Tokenizer import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cupy as cp
import pandas as pd
import numpy as np
import os
import pickle
import joblib
import json

def create_subsample(X, classes, size):
  size_per_class = size // len(classes)
  subsample = pd.DataFrame(columns=X.columns)

  for c in classes:
    shuffle_sample = X[X['status'] == c].sample(frac=1, random_state=42)
    subsample = pd.concat([subsample, shuffle_sample[:size_per_class]])

  return subsample

def main():
  df = pd.read_csv('text_classification/data/restaurant_review.csv')
  df = df.dropna(axis=0)

  sub_sample = create_subsample(df, [0,1,2], 30000)


  tokenizer = Tokenizer(num_merges=2500, oov_token='<UNK>')
  tokenizer.load('text_classification/weights_and_biases/Tokenizer/merge.txt', 'text_classification/weights_and_biases/Tokenizer/vocab.txt')
    
  token_dataset = []
  for text in sub_sample['text']: token_dataset.append(tokenizer.encode(text))

  # label_encoder = LabelEncoder()
  # encoded_label = label_encoder.fit_transform(df['status'].values)

  train_dataset, test_dataset, train_label, test_label = train_test_split(token_dataset, sub_sample['status'], test_size=0.4, random_state=42)
  test_dataset, val_dataset, test_label, val_label = train_test_split(test_dataset, test_label, test_size=0.5, random_state=42)

  train_label = cp.eye(3)[train_label.to_numpy().astype(int)]
  test_label = cp.eye(3)[test_label.to_numpy().astype(int)]
  val_label = cp.eye(3)[val_label.to_numpy().astype(int)]

  model = Sequential([
    Embedding(output_shape=300, input_shape=2756, activation=Linear(), max_lenght=350, embedded=True),
    LSTM(output_shape=32, input_shape=(350, 300), activation=Tanh(), recurrent_activation=Sigmoid(), return_sequences=False, bias_initialize=False, truncated_step=350)
  ])
  model.add(Dense(output_shape=3, activation=Softmax()))
    
  model.compile(loss=CrossEntropy(), optimizer=GradientDescent())
  his = model.fit(train_dataset, train_label, batch_size=32, epochs=5, lr=0.1)
  model.save_weights('text_classification/weights_and_biases')

if __name__ == '__main__':
  main()
  
