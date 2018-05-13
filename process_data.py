import pickle
import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from docopt import docopt

np.random.seed(3306)


# loads data and split into 10 folds.
def build_data_cv(filename, label, sentences, vocab, cv=10, clean_string=True):
    with open(filename, 'r') as f:
        for line in f:
            rev = [line.strip()]
            if clean_string:
                orig_rev = clean_str(' '.join(rev))
            else:
                orig_rev = ' '.join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {'y': label,
                     'text': orig_rev,
                     'num_words': len(orig_rev.split()),
                     'split': np.random.randint(0, cv)}
            sentences.append(datum)


# get word matrix. W[i] is the vector for word indexed by i.
def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


# loads 300x1 word vectors from file.
def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)
            word = str(word, 'UTF-8')
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


# add random vectors of unknown words which are not in pre-trained vector file.
# if pre-trained vectors are not used, then initialize all words in vocab with random value.
def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


# clean data.
def clean_str(string):
    string = re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', string)
    string = re.sub(r'\'s', ' \'s', string) 
    string = re.sub(r'\'ve', ' \'ve', string) 
    string = re.sub(r'n\'t', ' n\'t', string) 
    string = re.sub(r'\'re', ' \'re', string) 
    string = re.sub(r'\'d', ' \'d', string) 
    string = re.sub(r'\'ll', ' \'ll', string) 
    string = re.sub(r',', ' , ', string) 
    string = re.sub(r'!', ' ! ', string) 
    string = re.sub(r'\(', ' \( ', string) 
    string = re.sub(r'\)', ' \) ', string) 
    string = re.sub(r'\?', ' \? ', string) 
    string = re.sub(r'\s{2,}', ' ', string)    
    return string.strip().lower()


# main function.
def main():
    args = docopt('''
        Usage:
            process_data.py <vectors_file>
        ''')

    print('############')
    print('process data')
    print('############')

    vectors_file = args['<vectors_file>']                   # pre-trained word vectors file
    data_folder = ['rt-polarity.neg', 'rt-polarity.pos']    # data files
    datafile = 'mr.p'                                       # save data and word vectors

    print('Loading Data...')
    sentences = []              # sentences processed
    vocab = defaultdict(float)  # vocabulary
    # process data
    build_data_cv(data_folder[0], 0, sentences, vocab, cv=10, clean_string=True)
    build_data_cv(data_folder[1], 1, sentences, vocab, cv=10, clean_string=True)
    np.random.shuffle(sentences)
    maxlen = np.max(pd.DataFrame(sentences)['num_words'])    # max length of sentences
    print('Data Loaded!')
    print('Number Of Sentences: ' + str(len(sentences)))
    print('Vocab Size: ' + str(len(vocab)))
    print('Max Sentence Length: ' + str(maxlen))

    print('Loading Vectors...')
    vectors = load_bin_vec(vectors_file, vocab)     # pre-trained vectors
    print('Vectors Loaded!')
    print('Words Already In Vectors: ' + str(len(vectors)))
    # add random vectors of words which are not in vocab.
    add_unknown_words(vectors, vocab)
    W, word_idx_map = get_W(vectors)    # vectors of all words and a map of words to ids
    rand_vecs = {}                      # random vectors of all words
    # vectors of words with random values
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)            # random vectors of all words which are related to ids

    # save sentences and vectors
    pickle.dump([sentences, W, W2, word_idx_map, vocab, maxlen], open(datafile, 'wb'))
    print('Dataset created!')


# entry point.
if __name__ == '__main__':
    main()
