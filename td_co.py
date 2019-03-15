# 给定aspect代表的单词，求句子中每个单词与aspect的cosine值，
# 计算句子中与aspect单词的cosine值在一个阈值（0.7）上的单词个数，
# 单词个数最多者为该句子的aspect（可能有多个aspect）。

import numpy as np
import deal_data
import string
from nltk.corpus import stopwords as pw
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize

stop_words = set(pw.words('english'))

sentences = deal_data.restaurants()
aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']

wordsList = np.load('data/words.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
index = list(range(len(wordsList)))
words_index = dict(zip(index, wordsList))
wordVectors = np.load('data/wordVectors.npy')
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
            td.append(corpus.tf_idf(data, corpus))
    tf_idf.append(td)


# cosine = []
count_s = []
aspect_cosine = []
for i in range(len(sents)):
    sentences_vector = []
    for w in sents[i]:
        w = w.lower()
        if w not in stop_words:
            try:
                word_index = list(words_index.keys())[list(words_index.values()).index(w)]
                sentences_vector.append(wordVectors[word_index])
            except ValueError:
                continue
    count = []
    for aspect in aspect_vector:
        word_aspect_cosine = []
        ci = 0
        for j in range(len(sentences_vector)):
            data_td = deal_data.cosine(aspect, sentences_vector[j]) * tf_idf[i][j]
            word_aspect_cosine.append(data_td)
            if data_td > 0.0025:
                ci = ci + 1
        count.append(ci)
        aspect_cosine.append(word_aspect_cosine)

    count_s.append(count)
    # # print(aspect_cosine)
    # cosine.append(aspect_cosine)

print(len(sents))
print(len(sentences))
print(len(count_s))
#
count_a = 0
for i in range(len(sents)):
    index = [j for j, data in enumerate(count_s[i]) if data == max(count_s[i])]
    label = set()
    for data in index:
        if aspects[data] in sentences[i]['aspectCategories']:
            label.add(1)
        else:
            label.add(0)
    if len(label) == 1 and list(label)[0] == 1:
        count_a = count_a + 1

print(count_a / len(sentences))

# 0.65 0.433311432325887
# 0.70 0.4819316688567674
# 0.66 0.46
# 0.68 0.47
# 0.70 0.1360052562417871
