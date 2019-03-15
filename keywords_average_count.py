# 给定aspect以及其对应的keywords，对于一个aspect，
# 计算句子中word与keywords的cosine值均值作为该单词的特征。
# 统计单词特征在一个阈值以上的单词个数作为属于aspect的单词个数
# 单词个数最多者为该句子的aspect（可能有多个aspect）

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

count_s = []
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
    count = []
    for data in aspect_keywords:
        word_cosine = []
        c = 0
        for word_vector in sentences_vector:
            label = set()
            for i in range(len(data)):
                if deal_data.cosine(data[i], word_vector) > 0.7:
                    label.add(1)
            if 1 in label:
                c = c + 1
        count.append(c)

    count_s.append(count)

for count in count_s:
    print(count)

count_a = 0
for i in range(len(sentences)):
    index = [j for j, data in enumerate(count_s[i][1:]) if data == max(count_s[i][1:])]
    label_max = set()
    for data in index:
        if aspects[data] in sentences[i]['aspectCategories']:
            label_max.add(1)
        else:
            label_max.add(0)
    if len(label_max) == 1 and list(label_max)[0] == 1:
        count_a = count_a + 1

print(count_a / len(sentences))

# 0.491609081934847
