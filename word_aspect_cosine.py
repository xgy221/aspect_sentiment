# 给定aspect代表的单词，求句子中每个单词与aspect的cosine值，
# 计算句子中与aspect单词的cosine值在一个阈值（0.7）上的单词个数，
# 单词个数最多者为该句子的aspect（可能有多个aspect）。

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
# for aspect in aspect_vector:
#     print(aspect)

# print(deal_data.cosine([1,1],[0,1]))

# word_index = wordsList.index('ambience')
# print(wordVectors[word_index])

# cosine = []
count_s = []
for s in sentences:
    tran_tab = str.maketrans({key: None for key in string.punctuation})
    ss = s['text'].translate(tran_tab)
    s_w = ss.split()
    sentences_vector = []
    aspect_cosine = []
    for w in s_w:
        w = w.lower()
        if w not in stop_words:
            try:
                word_index = list(words_index.keys())[list(words_index.values()).index(w)]
                sentences_vector.append(wordVectors[word_index])
            except ValueError:
                continue
    count = [s['id']]
    for aspect in aspect_vector:
        word_aspect_cosine = []
        i = 0
        for word_vector in sentences_vector:
            word_aspect_cosine.append(deal_data.cosine(aspect, word_vector))
            if deal_data.cosine(aspect, word_vector) > 0.70:
                i = i + 1
        count.append(i)
        aspect_cosine.append(word_aspect_cosine)

    count_s.append(count)
    # print(aspect_cosine)
    # cosine.append(aspect_cosine)
#
for data in count_s:
    print(data)

count_a = 0
for i in range(len(sentences)):
    index = count_s[i].index(max(count_s[i]))
    if aspects[index] in sentences[i]['aspectCategories']:
        count_a = count_a + 1

print(count_a / len(sentences))

# 0.65 0.433311432325887
# 0.70 0.4819316688567674
# 0.66 0.46
# 0.68 0.47
