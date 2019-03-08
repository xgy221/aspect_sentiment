import numpy as np

wordsList = np.load('data/words_840B_300.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('data/wordVectors_840B_300.npy')
print('Loaded the word vectors!')

list(map(float, wordsList))