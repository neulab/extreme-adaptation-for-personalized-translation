from __future__ import print_function, division

import numpy as np
from collections import defaultdict

import sys

def append_to_file(sentences, filename):
    with open(filename, 'a+') as f:
        for s in sentences:
            print(s, file=f)

def read_talk(file, dic):
    talks = []
    with open(file, 'r') as f:
        for l in f:
            talks.append(dic[l.strip()])
    return talks


def read_corpus(file, dic, raw=False):
    """Read corpus in list of sentences

    Each sentence is a list of integers (determined by dic)

    Args:
        file (str): Corpus file path
        dic (defaultdict): Dictionary for the str -> int conversion

    Returns:
        Corpus
        list
    """
    sentences = []
    with open(file, 'r') as f:
        for l in f:
            if raw:
                sentences.append(l.strip().split())
            else:
                sent = [dic['SOS']]
                for w in l.split():
                    if w not in dic:
                        sent.append(dic['UNK'])
                    else:
                        sent.append(dic[w])
                sent.append(dic['EOS'])
                sentences.append(sent)
    return sentences


def read_user_data(list_file, dic):
    """Read corpus in list of sentences

    Each sentence is a list of integers (determined by dic)

    Args:
        file (str): Corpus file path
        dic (defaultdict): Dictionary for the str -> int conversion

    Returns:
        Corpus
        list
    """
    files = np.loadtxt(list_file, delimiter='\n', dtype=str)
    sentences = []
    for f in files:
        lines = []
        with open(f, 'r') as f:
            for l in f:
                sent = [dic['SOS']]
                for w in l.split():
                    if w not in dic:
                        sent.append(dic['UNK'])
                    else:
                        sent.append(dic[w])
                sent.append(dic['EOS'])
                lines.append(sent)
        sentences.append(lines)
    return sentences

def load_word_vectors(filename, dic):
    print('Reading word vectors from %s' % filename)
    non_zero = 0
    with open(filename, 'r') as f:
        # Read vector dimension in first line
        dim = int(f.readline().split()[1])
        vec = np.zeros((len(dic), dim))
        for l in f:
            word = l.split()[0].lower()
            if word in dic:
                non_zero += 1
                vector = np.asarray(l.split()[1:], dtype=float)
                vec[dic[word]] = vector
    print('Loaded %d pretrained word vectors (%.2f%%)' % (non_zero, 100 * non_zero / len(dic)))
    return vec

class User(object):

    def __init__(self, index):
        self.i = index
        self.sentence_pairs = []

class Batch(object):

    def __init__(self, src, trg, usr):
        self.src, self.trg, self.usr = src, trg, usr


class BatchLoader(object):
    """Iterator used to load batches

    Batches are predetermined so that each batch has only source sentence
    of the same length (easier for minibatching)
    """

    def __init__(self, datas, datat, datausr, bsize):
        """Constructor

        Args:
            datas (list): Source corpus
            datat (list): Target corpus
            bsize (int): Batch size
        """
        self.batches = []

        self.bs = bsize

        # Bucket samples by source sentence length
        buckets = defaultdict(list)
        users = {}
        for src, trg, usr in zip(datas, datat, datausr):
            if usr not in users:
                users[usr] = User(usr)
            users[usr].sentence_pairs.append((src, trg))
            buckets[len(src)].append((src, trg, usr))

        for src_len, bucket in buckets.items():
            np.random.shuffle(bucket)
            num_batches = int(np.ceil(len(bucket) * 1.0 / self.bs))
            for i in range(num_batches):
                cur_batch_size = self.bs if i < num_batches - 1 else len(bucket) - self.bs * i
                self.batches.append(([bucket[i * self.bs + j][0] for j in range(cur_batch_size)],
                                     [bucket[i * self.bs + j][1] for j in range(cur_batch_size)],
                                     [bucket[i * self.bs + j][2] for j in range(cur_batch_size)]))

        self.n = len(self.batches)
        self.reseed()

    def reseed(self):
        """Reshuffle the batches

        """
        print('Reseeding the dataset')
        self.i = 0
        np.random.shuffle(self.batches)

    def next(self):
        """Get next batch

        Returns:
            (source batch, target batch)
            tuple

        Raises:
            StopIteration: When all batches have been seen. Also resshuffles the batches
        """
        if self.i >= self.n:
            self.reseed()
            raise StopIteration()
        batch = Batch(*(self.batches[self.i]))
        self.i += 1
        return batch

    def __next__(self):
        """Same as self.next
        """
        return self.next()

    def __iter__(self):
        return self
