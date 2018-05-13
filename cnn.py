import os
import pickle
import time
import sys
import numpy as np
import tensorflow as tf
from docopt import docopt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# transforms sentence into a list of indices.
def get_idx_from_sent(sent, word_idx_map, maxlen, padding):
    x = []
    for i in range(padding):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < maxlen + 2 * padding:
        x.append(0)
    return x


# process datasets as 10-fold validation.
def make_idx_data_cv(sentences, word_idx_map, cv, maxlen, padding):
    train, test = [], []
    for sen in sentences:
        s = get_idx_from_sent(sen['text'], word_idx_map, maxlen, padding)
        s.append(sen['y'])
        if sen['split'] == cv:
            test.append(s)
        else:
            train.append(s)
    train = np.array(train, dtype='int')
    test = np.array(test, dtype='int')
    return [train, test]


# train each.
def train_cv(datasets,
             weights,
             W,
             maxlen,
             embedding_dims=300,
             dropoutd=0.5,
             batch_size=50,
             nb_epoch=12,
             nb_filter=100,
             filter_length=[3, 4, 5],
             hidden_dim=2,
             data_split=0,
             ngram=1,
             ):
    print('CV: ' + str(data_split + 1))

    np.random.seed(3306)
    tf.set_random_seed(3306)

    data_train = datasets[0]
    data_test = datasets[1]

    X_train = np.asarray([d[:-1] for d in data_train])
    Y_train = np.asarray([d[-1] for d in data_train])
    X_test = np.asarray([d[:-1] for d in data_test])
    Y_test = np.asarray([d[-1] for d in data_test])

    def softmaxY(Y):
        newY = []
        for y in Y:
            tmpY = [0] * hidden_dim
            tmpY[y] = 1
            newY.append(tmpY)
        return np.asarray(newY)

    Y_train = softmaxY(Y_train)
    Y_test = softmaxY(Y_test)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    # initialize certain positions of a filter.
    def set_weights(ws):
        maxweight = np.max(np.abs(weights))
        weight_norm = 1.0 * np.max(np.abs(ws)) / maxweight  # normalize pre-trained weights
        for i in range(nb_filter):
            for j in range(embedding_dims):
                for n in range(ngram):
                    if ngram < 3:
                        ws[n + 1][j][i] = weights[i][j + embedding_dims * n] * weight_norm
                    else:
                        ws[n][j][i] = weights[i][j + embedding_dims * n] * weight_norm
        return ws

    # initialize W in CNN.
    def conv_weight_variable(shape):
        initial = np.random.uniform(-0.01, 0.01, shape)
        initial = set_weights(initial)
        conv_W = tf.Variable(initial, name='conv_W', dtype=tf.float32)
        return conv_W

    # initialize bias in CNN.
    def conv_bias_variable(shape):
        initial = np.zeros(shape)
        conv_b = tf.Variable(initial, name='conv_b', dtype=tf.float32)
        return conv_b

    # initialize W in fully connected layer.
    def fcl_weight_variable(shape):
        initial = np.random.normal(0, 0.01, shape)
        fcl_W = tf.Variable(initial, name='fcl_W', dtype=tf.float32)
        return fcl_W

    # initialize bias in fully connected layer.
    def fcl_bias_variable(shape):
        initial = np.zeros(shape)
        fcl_b = tf.Variable(initial, name='fcl_b', dtype=tf.float32)
        return fcl_b

    # compute convolution.
    def conv1d(x, conv_W, conv_b):
        conv = tf.nn.conv1d(x,
                            conv_W,
                            stride=1,
                            padding='SAME',
                            name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu')
        return h

    # max-pooling.
    def max_pool(x):
        return tf.reduce_max(x, axis=1)

    # set all states to default.
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    # input.
    x = tf.placeholder(tf.int32, [None, maxlen], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, hidden_dim], name='input_y')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    # embedding.
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        embedding_table = tf.Variable(W, name='embedding_table')
        embedded_words = tf.nn.embedding_lookup(embedding_table, x)

    # CNN.
    pooled_outputs = []
    for i in filter_length:
        with tf.name_scope('conv_maxpool_%s' % i):
            filter_shape = [i, embedding_dims, nb_filter]
            conv_W = conv_weight_variable(filter_shape)
            conv_b = conv_bias_variable([nb_filter])
            conv = conv1d(embedded_words, conv_W, conv_b)
            pooled = max_pool(conv)
            pooled_outputs.append(pooled)

    nb_filter_total = nb_filter * len(filter_length)
    h_pool = tf.concat(pooled_outputs, 1)
    h_pool_flat = tf.reshape(h_pool, [-1, nb_filter_total])

    # dropout.
    with tf.name_scope('dropout'):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob) * (1 - dropout_keep_prob + dropoutd)

    # fully connected layer.
    with tf.name_scope('fcl'):
        fcl_W = fcl_weight_variable([nb_filter_total, hidden_dim])
        fcl_b = fcl_bias_variable([hidden_dim])
        fcl_output = tf.matmul(h_drop, fcl_W) + fcl_b
        y = tf.nn.softmax(fcl_output)

    # loss.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # optimizer
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08)
    train_step = optimizer.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train.
    tf.global_variables_initializer().run()

    test_accuracy = 0
    for e in range(nb_epoch):
        epoch_starttime = time.time()
        i = 0
        while i < len(X_train):
            if i + batch_size < len(X_train):
                batch_xs = X_train[i:i + batch_size]
                batch_ys = Y_train[i:i + batch_size]
            else:
                batch_xs = X_train[i:]
                batch_ys = Y_train[i:]
            i += batch_size
            train_step.run({x: batch_xs, y_: batch_ys, dropout_keep_prob: dropoutd})
        train_loss = sess.run(cross_entropy,
                              feed_dict={x: X_train[-10 * batch_size:],
                                         y_: Y_train[-10 * batch_size:],
                                         dropout_keep_prob: 1.0})
        test_accuracy = sess.run(accuracy,
                                 feed_dict={x: X_test,
                                            y_: Y_test,
                                            dropout_keep_prob: 1.0})

        sys.stdout.write('Epoch: %d' % (e+1))
        sys.stdout.write('\tTrain Loss: %.6f' % train_loss)
        sys.stdout.write('\tEpoch Time: %.1fs' % (time.time()-epoch_starttime))
        sys.stdout.write('\n')

    print('CV: ' + str(data_split + 1) + ' Test Accuracy: %.4f%%\n' % (100 * test_accuracy))

    sess.close()

    return test_accuracy


# main function
def main():
    args = docopt('''
            Usage:
                cnn.py [options] <data_path> <weights_path>

            Options:
                --ngram NUM         N-gram: uni-gram=1, bi-gram=2, tri-gram=3 [default: 1]
                --padding NUM       pad a sentence with 0 in both sides [default: 4]
            ''')

    print('#########')
    print('Train CNN')
    print('#########')

    data_path = args['<data_path>']
    weights_path = args['<weights_path>']
    ngram = int(args['--ngram'])
    padding = int(args['--padding'])

    print('Loading Data...')
    data_file = open(data_path, 'rb')
    x = pickle.load(data_file)
    data_file.close()
    sentences, W, W2, word_idx_map, vocab, maxlen = x[0], x[1], x[2], x[3], x[4], x[5]
    print('Data Loaded!')
    print('Loading Weights...')
    weights_file = open(weights_path, 'rb')
    weights = pickle.load(weights_file)
    weights_file.close()
    print('Weights Loaded!')

    final = []
    for i in range(10):
        datasets = make_idx_data_cv(sentences, word_idx_map, i, maxlen, padding)
        acc = train_cv(datasets,
                       weights[i],
                       W,
                       data_split=i,
                       maxlen=maxlen + 2 * padding,
                       ngram=ngram)
        final.append(acc)

    print('Final Test Accuracy:' + str(np.mean(final)))


# entry point.
if __name__ == '__main__':
    main()
