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

count_sum = []
i = 0

for s in sentences:
    for i in s['text']:
        if i in string.punctuation:  # 如果字符是标点符号的话就将其替换为空格
            s['text'] = s['text'].replace(i, " ")
    s_w = s['text'].split()
    sentences_vector = []
    count_s = [s['id']]
    for w in s_w:
        w = w.lower()
        if w not in stop_words:
            try:
                word_index = list(words_index.keys())[list(words_index.values()).index(w)]
                sentences_vector.append(wordVectors[word_index])
            except ValueError:
                continue
    for data in aspect_keywords:
        count = 0
        for word_vector in sentences_vector:
            for i in range(len(data)):
                if deal_data.cosine(data[i], word_vector) > 0.70:
                    count = count + 1
                    break
        count_s.append(count)
    count_sum.append(count_s)

for count_s in count_sum:
    print(count_s)

# count_a = 0
# for i in range(len(sentences)):
#     index = count_sum[i].index(max(count_sum[i][1:]))
#     if aspects[index] in sentences[i]['aspectCategories']:
#         count_a = count_a + 1
#
# print(count_a / len(sentences))




