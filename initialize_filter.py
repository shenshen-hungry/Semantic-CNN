from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle
import math
import operator
import re
import numpy as np
from docopt import docopt

np.random.seed(3306)


# transforms sentence into a list of indices.
def get_idx_from_sent(sent, word_idx_map, maxlen, padding):
    x = []
    for i in xrange(padding):
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
        s = get_idx_from_sent(sen["text"], word_idx_map, maxlen, padding)
        s.append(sen["y"])
        if sen["split"] == cv:
            test.append(s)
        else:
            train.append(s)
    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    return [train, test]


# using k-means to get clusters of N-grams.
def get_cluster(vecs, nb_filter, embedding_size, ngram):
    cluster = [[vecs[-1]]]
    while len(cluster) < nb_filter / 2:
        d2 = np.array([min([np.linalg.norm(v-c[0])**2 for c in cluster]) for v in vecs])
        probs = d2 / d2.sum()
        cumprobs = probs.cumsum()
        r = np.random.random()
        ind = np.where(cumprobs >= r)[0][0]
        cluster.append([vecs[ind]])
    for i in xrange(20):
        for v in vecs:
            min_val = 100
            min_pos = 0
            for c in xrange(len(cluster)):
                dis = np.linalg.norm(v - cluster[c][0])
                if dis < min_val:
                    min_val = dis
                    min_pos = c
            cluster[min_pos].append(v)
        for c in xrange(len(cluster)):
            s = [0] * embedding_size * ngram
            for v in cluster[c][1:]:
                s += v
            if len(cluster[c][1:]) > 0:
                for v in xrange(len(s)):
                    s[v] /= len(cluster[c][1:])
                cluster[c] = [s]
    return cluster


# get weights which are used to initialize CNN.
def get_weights(datasets,
                W,
                maxlen,
                data_split=0,
                nb_filter=100,
                embedding_size=300,
                ngram=2,
                ):
    print("CV: " + str(data_split + 1) + "/10")
    print("Initializing Filters...")

    # get N-grams.
    pos = {}
    neg = {}
    for l in datasets[0]:
        for w in xrange(0, maxlen):
            ws = ""
            p = w
            isnull = True
            while p >= 0 and p > w - ngram:
                if l[p] != 0:
                    isnull = False
                if ws == "":
                    ws = str(l[p])
                else:
                    ws = str(l[p]) + "," + ws
                p -= 1
            if isnull:
                continue
            if ws == "":
                continue
            if l[-1] == 1:
                if ws in pos:
                    pos[ws] += 1
                else:
                    pos[ws] = 1
            else:
                if ws in neg:
                    neg[ws] += 1
                else:
                    neg[ws] = 1

    # arrange N-grams.
    pos_ratio = {}
    neg_ratio = {}
    # to balance pos and neg.
    pos_num = np.sum(pos.values())
    neg_num = np.sum(neg.values())
    pos_neg_ratio = 1.0 * pos_num / neg_num
    # compute the priority of each N-gram.
    for k in pos.keys():
        if pos[k] > 1:
            if k in neg:
                pos_ratio[k] = (1.0 / pos_neg_ratio * pos[k] + 1) / (neg[k] + 1) * math.log(
                    max(1.0 / pos_neg_ratio * pos[k] - neg[k], 1))
            else:
                pos_ratio[k] = (1.0 / pos_neg_ratio * pos[k] + 1) / 1 * math.log(
                    1.0 / pos_neg_ratio * pos[k])
    for k in neg.keys():
        if neg[k] > 1:
            if k in pos:
                neg_ratio[k] = (1.0 * pos_neg_ratio * neg[k] + 1) / (pos[k] + 1) * math.log(
                    max(1.0 * pos_neg_ratio * neg[k] - pos[k], 1))
            else:
                neg_ratio[k] = (1.0 * pos_neg_ratio * neg[k] + 1) / 1 * math.log(
                    1.0 * pos_neg_ratio * neg[k])
    pos_sorted = sorted(pos_ratio.items(), key=operator.itemgetter(1))
    neg_sorted = sorted(neg_ratio.items(), key=operator.itemgetter(1))

    # choose the top 10% N-grams.
    def choose_ngrams(ngram_sorted):
        vecs = []
        for s in ngram_sorted[-int(0.1 * len(ngram_sorted)):]:
            vec_tmp = []
            for w in s[0].split(','):
                if len(vec_tmp) == 0:
                    vec_tmp = W[int(w)]
                else:
                    vec_tmp = np.concatenate((vec_tmp, W[int(w)]), axis=0)
            vecs.append(vec_tmp)
        return vecs
    pos_vec = choose_ngrams(pos_sorted)
    neg_vec = choose_ngrams(neg_sorted)

    # get clusters from N-grams.
    pos_cluster = get_cluster(pos_vec, nb_filter, embedding_size, ngram)
    neg_cluster = get_cluster(neg_vec, nb_filter, embedding_size, ngram)

    # save weights.
    conv_weights = []
    for i in xrange(int(nb_filter / 2)):
        p = pos_cluster[i]
        conv_weights.append([[v] for v in p[0]])
    for i in xrange(int(nb_filter / 2)):
        n = neg_cluster[i]
        conv_weights.append([[v] for v in n[0]])

    print("Filters Initialized!")
    return conv_weights


# main function.
def main():
    args = docopt("""
        Usage:
            initialize_filter.py [options] <data_path>

        Options:
            --ngram NUM         N-gram: uni-gram=1, bi-gram=2, tri-gram=3 [default: 1]
            --padding NUM       pad a sentence with 0 in both sides [default: 4]
        """)

    print("#################")
    print("initialize filter")
    print("#################")

    data_path = args['<data_path>']
    ngram = int(args['--ngram'])
    padding = int(args['--padding'])

    print('Loading Data...')
    data_file = open(data_path, 'r')
    x = cPickle.load(data_file)
    data_file.close()
    sentences, W, W2, word_idx_map, vocab, maxlen = x[0], x[1], x[2], x[3], x[4], x[5]
    print("Data Loaded!")

    w = []
    for i in xrange(0, 10):
        datasets = make_idx_data_cv(sentences, word_idx_map, i, maxlen, padding)
        w.append(get_weights(datasets, W, maxlen + 2 * padding, data_split=i, ngram=ngram))
    wdata = open("weights_" + str(ngram) + ".pkl", "w")
    cPickle.dump(w, wdata)
    wdata.close()

# entry point.
if __name__ == '__main__':
    main()
