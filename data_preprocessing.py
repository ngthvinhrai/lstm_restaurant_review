from NeuralNetworks.Tokenizer import Tokenizer
import pandas as pd
import numpy as np
import cupy as cp
import re

def create_subsample(X, classes, size):
    size_per_class = size // len(classes)
    subsample = pd.DataFrame(columns=X.columns)

    for c in classes:
        shuffle_sample = X[X['status'] == c].sample(frac=1, random_state=42)
        subsample = pd.concat([subsample, shuffle_sample[:size_per_class]])

    return subsample

def padding(X, max_length):
    if max_length < 256: b = cp.int8
    elif max_length < 65536: b = cp.int16
    new_X = cp.zeros((len(X), max_length), dtype=b)

    for x, newx in zip(X, new_X):
        newx[(max_length-len(x)):] = cp.array([x], dtype=b)
    
    return new_X

if __name__ == '__main__':
    df = pd.read_csv('lstm_restaurant_review/data/restaurant_review.csv')
    df = df.dropna(axis=0)

    sub_sample = create_subsample(df, [0,1,2], 30000)

    tokenizer = Tokenizer(num_merges=2500, oov_token='<UNK>')
    tokenizer.load('lstm_restaurant_review/weights_and_biases/Tokenizer/merge.txt', 'lstm_restaurant_review/weights_and_biases/Tokenizer/vocab.txt')
    
    token_dataset = []
    for text in sub_sample['text']: token_dataset.append(tokenizer.encode(text))

    max_length = 350

    token_dataset = padding(token_dataset, 350)

    numpy_class = df['status'].to_numpy(dtype=cp.int8)

    cp.save('lstm_restaurant_review/data/X.npy', token_dataset)
    cp.save('lstm_restaurant_review/data/Y.npy', numpy_class)
