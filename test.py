import deal_data
import numpy as np
import nltk

# nltk.download('stopwords')

# sentences = deal_data.restaurants()
# aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']
#
# print(len(sentences))
#
# i = 0
# j = 0
# for s in sentences:
#     if aspects[1] in s['aspectCategories']:
#         i = i + 1
#     if aspects[2] in s['aspectCategories']:
#         j = j + 1
#
# print(i)
# print(j)
#
# vector = [0 for _ in range(50)]
# a = [1, 2, 3]
# b = [1, 1, 1]
# print(a+b)
#
# print(deal_data.list_add(a, b))
#
# d = [c**2 for c in a]
# print(d)
#
# wordsList = np.load('data/words.npy')
# print('Loaded the word list!')
# wordsList = wordsList.tolist()  # Originally loaded as numpy array
# wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
# wordVectors = np.load('data/wordVectors.npy')
# print('Loaded the word vectors!')
#
#
# aspect_keywords = []
# with open('data/keywords.txt') as key:
#     for line_data in key:
#         line = line_data.split()
#         keywords_vector = []
#         for word in line[1:]:
#             aspect_index = wordsList.index(word)
#             keywords_vector.append(wordVectors[aspect_index])
#         aspect_keywords.append(keywords_vector)
#
# print(len(aspect_keywords))
#
# for data in aspect_keywords:
#     print(data)
#
from nltk.corpus import stopwords as pw
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize

stop_words = set(pw.words('english'))

sentence = ['but', 'the', 'staff', 'was', 'so', 'horrible', 'to', 'us']
filtered_sentence = []

for w in sentence:
    if w not in stop_words:
        filtered_sentence.append(w)

print(sentence)
print(filtered_sentence)

import numpy as np
import deal_data
from nltk.corpus import stopwords as pw
import string

stop_words = set(pw.words('english'))

sentences = deal_data.restaurants()

wordsList = np.load('data/words.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
index = list(range(len(wordsList)))
words_index = dict(zip(index, wordsList))
wordVectors = np.load('data/wordVectors.npy')
print('Loaded the word vectors!')

aspect_keywords = []
with open('data/keyword_test.txt') as key:
    for line_data in key:
        line = line_data.split()
        keywords_vector = []
        for word in line:
            if len(word) < 10:
                aspect_index = list(words_index.keys())[list(words_index.values()).index(word)]
                keywords_vector.append(wordVectors[aspect_index])
            else:
                aspect_term = word.split('/')
                sum_term = 0
                for aspect_term1 in aspect_term:
                    aspect_index = list(words_index.keys())[list(words_index.values()).index(aspect_term1)]
                    sum_term = sum_term + wordVectors[aspect_index]
                keywords_vector.append(sum_term / len(aspect_term))
        aspect_keywords.append(keywords_vector)

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
        data = data.lower()
        if data not in stop_words:
            # print(data)
            td.append(corpus.tf_idf(data, corpus))
    tf_idf.append(td)

for aspect in aspect_keywords:
    for vector in aspect[1:]:
        print(deal_data.cosine(aspect[0], vector)*corpus.tf_idf('food', corpus))
    print('\n')
