import pandas as pd
import numpy as np
import re

class Tokenizer:
    def __init__(self, num_words, max_length, filter='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, oov_token=None):
        self.num_words = num_words
        self.max_lenght = max_length
        self.filter = filter
        self.lower = lower
        self.oov_token = oov_token
        self.word_counts = {}
        self.word_index = {}

    def fit(self, dataset):
        for text in dataset:
            for word in text.split():
                if word not in self.word_counts: self.word_counts[word] = 1
                else: self.word_counts[word] += 1 
                    
        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)

        if self.oov_token is None: sorted_wcounts = []
        else: sorted_wcounts = [self.oov_token]
        sorted_wcounts.extend(wc[0] for wc in wcounts)

        self.word_index = dict(zip(sorted_wcounts, range(1, len(sorted_wcounts)+1)))

    def texts_to_sequences(self, dataset, padding=False):
        ttsq = []

        for text in dataset:
            lenght = len(text.split())
            seq = []
            if padding == True:
                if lenght < self.max_lenght:
                    seq.extend([0 for _ in range(self.max_lenght - lenght)])
            
            for word in text.split():
                pos = self.word_index[word]
                if pos > self.num_words:
                    if self.oov_token is not None: seq.append(self.word_index[self.oov_token])
                else: seq.append(pos)
            ttsq.append(seq)

        return ttsq 

    def save_word_index(self, path):
        with open(path, 'w', encoding="utf-8") as file:
            for word, index in self.word_index.items():
                file.write(word + " " + str(index) + "\n")
    
    def load_word_index(self, path):
        with open(path, 'r', encoding="utf-8") as file:
            for line in file:
                text = line.strip().split()
                self.word_index[text[0]] = int(text[1])

class DataPreprocessing:
    def __init__(self, data):
        self.data = data

    def one_hot_processing(self):
        Y = self.data['Label'].to_numpy()
        Y = np.eye(3)[Y]

        X = self.data.drop('Label', axis=1).to_numpy()
        X = np.eye(np.max(X)+1)[X]

        return X, Y
    
    def binary_processing(self):
        Y = self.data['Label'].to_numpy()
        Y = np.eye(3)[Y]

        X = np.array([[[int(bit) for bit in format(num, '07b')] for num in row] for row in self.data.drop('Label', axis=1).to_numpy()])
        
        return X, Y

if __name__ == '__main__':
    df = pd.read_csv('test_model/data/IMDB_Dataset.csv')

    tokenizer = Tokenizer(num_words=5000, max_length=812)
    tokenizer.fit(df['review'].head())
    print(tokenizer.texts_to_sequences(df['review'].head(), padding=True))