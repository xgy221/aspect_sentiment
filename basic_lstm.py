import tensorflow as tf
import numpy as np
import datetime
import deal_data
import random
import string

sentences = deal_data.restaurants_train_new()

# 设置一些参数
batchSize = 24
units = 64
numClasses = 2
iterations = 1000
max_length = 20
num_dimensions = 50
d = 50
aspects = ['service', 'ambience', 'anecdotes', 'price', 'food']

# glove词向量处理
wordsList = np.load('data/words.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('data/wordVectors.npy')
print('Loaded the word vectors!')

# 构建LSTM网络
tf.reset_default_graph()

cv = locals()

def aspect_lstm(aspect):
    cv[aspect + '_s'] = tf.placeholder(tf.int32, [batchSize, max_length])
    cv[aspect + '_w'] = tf.placeholder(tf.int32, [batchSize, max_length])
    data = tf.nn.embedding_lookup(wordVectors, cv[aspect + '_s'])
    lstmCell = tf.nn.rnn_cell.LSTMCell(units)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    weight = tf.Variable(tf.truncated_normal([units, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    return value

sis = []
# 创建自调整变量
w = tf.get_variable(
    name='wad',
    shape=[numClasses, d],
    initializer=tf.random_uniform_initializer(-0.01, 0.01),
    regularizer=tf.contrib.layers.l2_regularizer(0.01)
)

for a in aspects:
    result = tf.nn.softmax('ch,bmh->bmc', w, aspect_lstm(a))

    labels = tf.placeholder(tf.float32, [batchSize, len(aspects), numClasses])
    correctPred = tf.equal(tf.argmax(result, 2), tf.argmax(labels, 2))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

# 建立tf会话，只有在会话中，tf才会具体计算值，包括各种占位符和变量
sess = tf.Session()

# 构造可视化图标变量
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

feedDict = {}

aspect_w = np.zeros([batchSize, max_length])
a_ids = np.zeros([max_length])
for a in aspects:
    a_w = a.split()
    w_i = 0
    for w in a_w:
        try:
            a_ids[w_i] = wordsList.index(w)
        except ValueError:
            a_ids[w_i] = 399999  # Vector for unkown words
        w_i += 1
for i in range(batchSize):
    aspect_w[i] = a_ids
for a in aspects:
    feedDict[cv[a + '_w']] = aspect_w

for i in range(iterations):
    ss = random.sample(sentences, batchSize)

    batchLabels = np.zeros([batchSize, 3])
    s_s = np.zeros([batchSize, max_length])
    s_i = 0
    for s in ss:
        for i in s['text']:
            if i in string.punctuation:  # 如果字符是标点符号的话就将其替换为空格
                s['text'] = s['text'].replace(i, " ")
        s_w = s['text'].split()
        s_ids = np.zeros([max_length])
        w_i = 0
        for w in s_w:
            w = w.lower()
            try:
                s_ids[w_i] = wordsList.index(w)
            except ValueError:
                s_ids[w_i] = 399999  # Vector for unkown words
            w_i += 1
            if w_i >= 30:
                break
        s_s[s_i] = s_ids

        bl = []
        a_i = 0
        sentiment = s['aspectCategories'][a] if s['aspectCategories'].__contains__(a) else 0
        batchLabels[s_i][sentiment] = 1
        feedDict[cv[a + '_s']] = s_s
        s_i += 1
    feedDict[labels] = batchLabels

    # Write summary to Tensorboard
    if i % 50 == 0:
        summary = sess.run(merged, feedDict)
        writer.add_summary(summary, i)

    # Save the network every 10,000 training iterations
    if i % 10000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()
