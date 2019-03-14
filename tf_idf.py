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
    for data in sen:
        elem = []
        if data not in stop_words:
            td.append(corpus.tf_idf(data, corpus))
    tf_idf.append(td)

# print(tf_idf[0][3])

cosine = []
for i in range(len(sents)):
    sentences_vector = []
    average_cosine = []
    for w in sents[i]:
        w = w.lower()
        if w not in stop_words:
            try:
                word_index = list(words_index.keys())[list(words_index.values()).index(w)]
                sentences_vector.append(wordVectors[word_index])
            except ValueError:
                continue
    for aspect in aspect_vector:
        sum_cosine = 0
        for j in range(len(sentences_vector)):
            sum_cosine = sum_cosine + \
                         deal_data.cosine(aspect, sentences_vector[j]) * tf_idf[i][j]
        if len(sentences_vector) != 0:
            average_cosine.append(sum_cosine / len(sentences_vector))
    cosine.append(average_cosine)

count = 0
count_empty = 0
for i in range(len(sentences)):
    if len(cosine[i]) == 0:
        count_empty = count_empty + 1
    else:
        index = cosine[i].index(max(cosine[i]))
        if aspects[index - 1] in sentences[i]['aspectCategories']:
            count = count + 1
print(count_empty)
print(count / (len(sentences) - count_empty))
