import deal_data
import numpy as np

sentences = deal_data.restaurants()
aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']

print(len(sentences))

i = 0
j = 0
for s in sentences:
    if aspects[1] in s['aspectCategories']:
        i = i + 1
    if aspects[2] in s['aspectCategories']:
        j = j + 1

print(i)
print(j)

vector = [0 for _ in range(50)]
a = [1, 2, 3]
b = [1, 1, 1]
print(a+b)

print(deal_data.list_add(a, b))

d = [c**2 for c in a]
print(d)

wordsList = np.load('data/words.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('data/wordVectors.npy')
print('Loaded the word vectors!')


aspect_keywords = []
with open('data/keywords.txt') as key:
    for line_data in key:
        line = line_data.split()
        keywords_vector = []
        for word in line[1:]:
            aspect_index = wordsList.index(word)
            keywords_vector.append(wordVectors[aspect_index])
        aspect_keywords.append(keywords_vector)

print(len(aspect_keywords))

for data in aspect_keywords:
    print(data)