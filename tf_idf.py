from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
import deal_data
import numpy as np
import string
from nltk.corpus import stopwords as pw

stop_words = set(pw.words('english'))

sentences = deal_data.restaurants()
aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']

wordsList = np.load('data/words_840B_300.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
index = list(range(len(wordsList)))
words_index = dict(zip(index, wordsList))
wordVectors = np.load('data/wordVectors_840B_300.npy')
print('Loaded the word vectors!')

aspect_vector = []
for aspect in aspects:
    if len(aspect) < 10:
        aspect_index = list(words_index.keys())[list(words_index.values()).index(aspect)]
        aspect_vector.append(wordVectors[aspect_index])
    else:
        aspect_term = aspect.split('/')
        sum_term = 0
        for aspect_term1 in aspect_term:
            aspect_index = list(words_index.keys())[list(words_index.values()).index(aspect_term1)]
            sum_term = sum_term + wordVectors[aspect_index]
        aspect_vector.append(sum_term / len(aspect_term))

sentence = []
for s in sentences:
    for i in s['text']:
        if i in string.punctuation:  # 如果字符是标点符号的话就将其替换为空格
            s['text'] = s['text'].replace(i, " ")
    sentence.append(s['text'])

sents = [word_tokenize(sent) for sent in sentence]

corpus = TextCollection(sents)

tf_idf = []
for sen in sents:
    td = []
    word = []
    for data in sen:
        if data not in stop_words:
            word.append(data)
            td.append(corpus.tf_idf(data, corpus))
    words_index = dict(zip(word, td))
    print(words_index)
    tf_idf.append(words_index)


