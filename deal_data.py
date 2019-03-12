import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import os


def deal_glove():
    """
    处理glove, 生成中间变量
    """
    words = []
    wordVectors = []

    count = 0

    with open('data/glove.840B.300d.txt') as glove:
        for line in glove:
            line = line.split()
            if len(line) == 301:
                words.append(line[0].encode('utf-8'))
                wordVectors.append(list(map(float, line[1:])))
            else:
                count = count + 1
    # 存储结果
    print(count)
    words = np.array(words)
    np.save('data/words_840B_300.npy', words)
    wordVectors = np.array(wordVectors, dtype=np.float32)
    np.save('data/wordVectors_840B_300.npy', wordVectors)
    # for vector in wordVectors:
    #     if len(vector) != 300:
    #         print(wordVectors.index(vector))


def restaurants():
    dom_tree = xml.dom.minidom.parse(
        os.path.split(os.path.realpath(__file__))[0] + "/SemEval2014/Restaurants_Train.xml")

    sentences = list()
    _sentences = dom_tree.getElementsByTagName('sentence')
    for sentence in _sentences:
        _aspectCategories = sentence.getElementsByTagName('aspectCategory')
        aspectCategories = []
        for aspectCategory in _aspectCategories:
            aspectCategories.append(aspectCategory.getAttribute('category'))
        sentences.append({
            'id': sentence.getAttribute('id'),
            'text': sentence.getElementsByTagName('text')[0].firstChild.nodeValue,
            'aspectCategories': aspectCategories
        })
    return sentences


def cosine(aspect, word):
    aspect = np.mat(aspect)
    word = np.mat(word)
    num = float(aspect * word.T)
    de_nom = np.linalg.norm(aspect) * np.linalg.norm(word)
    cos = num / de_nom  # 余弦值
    sim = 0.5 + 0.5 * cos
    return sim


def list_add(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c


def count_cosine(a):
    count = 0
    for i in a:
        if i < 0.7:
            count = count + 1
    return count

# deal_glove()
