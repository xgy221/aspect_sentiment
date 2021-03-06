# 给定aspect以及其对应的keywords，对于一个aspect，
# 计算句子中word与keywords的cosine值均值作为该单词的特征。
# 再计算句子中所有word的均值作为句子的特征，取其中最大者作为句子的aspect。

import numpy as np
import deal_data
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

aspect_keywords = []
with open('data/keywords.txt') as key:
    for line_data in key:
        line = line_data.split()
        keywords_vector = []
        for word in line[1:]:
            aspect_index = list(words_index.keys())[list(words_index.values()).index(word)]
            keywords_vector.append(wordVectors[aspect_index])
        aspect_keywords.append(keywords_vector)
# for aspect in aspect_vector:
#     print(aspect)

# print(deal_data.cosine([1,1],[0,1]))

# word_index = wordsList.index('ambience')
# print(wordVectors[word_index])

cosine = []
for s in sentences:
    for i in s['text']:
        if i in string.punctuation:  # 如果字符是标点符号的话就将其替换为空格
            s['text'] = s['text'].replace(i, " ")
    s_w = s['text'].split()
    sentences_vector = []
    average_cosine = [s['id']]
    for w in s_w:
        w = w.lower()
        if w not in stop_words:
            try:
                word_index = list(words_index.keys())[list(words_index.values()).index(w)]
                sentences_vector.append(wordVectors[word_index])
            except ValueError:
                continue
    for data in aspect_keywords:
        word_cosine = []
        for word_vector in sentences_vector:
            sum_cosine = 0
            for i in range(len(data)):
                sum_cosine = sum_cosine + deal_data.cosine(data[i], word_vector)
            word_cosine.append(sum_cosine/len(data))
        if len(sentences_vector) != 0:
            average_cosine.append(sum(word_cosine) / len(word_cosine))
    cosine.append(average_cosine)

# print(cosine)

# 计算准确率accuracy:两个及以上aspect的句子怎么算？

# print(len(sentences))
# print(len(cosine))
# for i in range(len(sentences)):
#     print(cosine[i])
#     print(sentences[i])

count = 0
count_empty = 0
for i in range(len(sentences)):
    if len(cosine[i]) == 1:
        count_empty = count_empty + 1
    else:
        index = cosine[i].index(max(cosine[i][1:]))
        if aspects[index-1] in sentences[i]['aspectCategories']:
            count = count + 1

print(count/(len(sentences)-count_empty))

# 0.491609081934847
