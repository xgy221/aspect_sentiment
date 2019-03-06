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

# cosine = []
i = 0
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
                word_index = wordsList.index(w)
                sentences_vector.append(wordVectors[word_index])
            except ValueError:
                continue
    count = []
    for aspect in aspect_vector:
        word_aspect_cosine = []
        i = 0
        for word_vector in sentences_vector:
            word_aspect_cosine.append(deal_data.cosine(aspect, word_vector))
            if deal_data.cosine(aspect, word_vector) > 0.80:
                i = i + 1
        count.append(i)
        aspect_cosine.append(word_aspect_cosine)

    count_s.append(count)
    # print(aspect_cosine)
    # cosine.append(aspect_cosine)
#
# for data in cosine:
#     print(data)
# print(len(count_s))

count_a = 0
for i in range(len(sentences)):
    index = count_s[i].index(max(count_s[i]))
    if aspects[index] in sentences[i]['aspectCategories']:
        count_a = count_a + 1

print(count_a/len(sentences))