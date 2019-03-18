import tensorflow as tf
import numpy as np
import datetime

# 设置一些参数
batchSize = 24
units = 64
numClasses = 2
iterations = 100000
max_length = 20
num_dimensions = 50

# glove词向量处理
wordsList = np.load('data/words.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('data/wordVectors.npy')
print('Loaded the word vectors!')

# 构建LSTM网络
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, max_length])

# data = tf.Variable(tf.zeros([batchSize, max_length, num_dimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.nn.rnn_cell.LSTMCell(units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([units, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# 加载模型
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

import re

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def get_sentence_matrix(sentence):
    arr = np.zeros([batchSize, max_length])
    sentenceMatrix = np.zeros([batchSize, max_length], dtype='int32')
    cleanedSentence = clean_sentences(sentence)
    split = cleanedSentence.split()
    for indexCounter, word in enumerate(split):
        try:
            sentenceMatrix[0, indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0, indexCounter] = 399999  # Vector for unkown words
    return sentenceMatrix


inputText = "That movie was terrible."
inputMatrix = get_sentence_matrix(inputText)
predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# predictedSentiment[0] represents output score for positive sentiment
# predictedSentiment[1] represents output score for negative sentiment

if (predictedSentiment[0] > predictedSentiment[1]):
    print("Positive Sentiment")
else:
    print("Negative Sentiment")