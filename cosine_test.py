import numpy as np
import deal_data
import string
from nltk.corpus import stopwords as pw

stop_words = set(pw.words('english'))

sentences = deal_data.restaurants()
aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']

wordsList = np.load('data/words.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('data/wordVectors.npy')
print('Loaded the word vectors!')

aspect_vector = []
for aspect in aspects:
    if len(aspect) < 10:
        aspect_index = wordsList.index(aspect)
        aspect_vector.append(wordVectors[aspect_index])
    else:
        aspect_term = aspect.split('/')
        sum_term = 0
        for aspect_term1 in aspect_term:
            aspect_index = wordsList.index(aspect_term1)
            sum_term = sum_term + wordVectors[aspect_index]
        aspect_vector.append(sum_term / len(aspect_term))
# for aspect in aspect_vector:
#     print(aspect)

# print(deal_data.cosine([1,1],[0,1]))

# word_index = wordsList.index('ambience')
# print(wordVectors[word_index])

cosine = []
i = 0
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
                word_index = wordsList.index(w)
                sentences_vector.append(wordVectors[word_index])
            except ValueError:
                continue
    for aspect in aspect_vector:
        sum_cosine = 0
        for word_vector in sentences_vector:
            sum_cosine = sum_cosine + deal_data.cosine(aspect, word_vector)
        if len(sentences_vector) != 0:
            average_cosine.append(sum_cosine / len(sentences_vector))
    cosine.append(average_cosine)

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
print(count_empty)
print(count/(len(sentences)-count_empty))
