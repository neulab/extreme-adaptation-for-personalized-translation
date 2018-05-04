from __future__ import division, print_function

import numpy as np
import dynet as dy

from collections import defaultdict
import pickle

class LanguageModel(object):
    def p_next(self, sent):
        pass

    def init(self):
        pass

    def p_next_expr(self, sent):
        return dy.inputTensor(self.p_next(sent))

    def fit(self, corpus):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class UniformLanguageModel(LanguageModel):
    def __init__(self, w2id):
        self.n = len(w2id)

    def p_next(self, sent):
        return np.ones(self.n) / self.n


class UnigramLanguageModel(LanguageModel):
    def __init__(self, w2id, eps=0):
        self.w2id = w2id
        self.eps = eps
        self.unigrams = np.ones(len(self.w2id)) / len(self.w2id)

    def init(self):
        self.u_e = dy.inputTensor(self.unigrams)

    def p_next(self, sent):
        return self.unigrams

    def p_next_expr(self, sent):
        return self.u_e

    def fit(self, corpus):
        self.unigrams = np.zeros(len(self.w2id)) + self.eps
        for sent in corpus:
            for w in sent:
                self.unigrams[w] += 1
        self.unigrams /= self.unigrams.sum()

    def save(self, filename):
        np.save(filename, self.unigrams)

    def load(self, filename):
        if not filename.endswith('.npy'):
            filename += '.npy'
        self.unigrams = np.load(filename)

def zero():
    return 0.0

def dd():
    return defaultdict(zero)

class BigramLanguageModel(LanguageModel):
    def __init__(self, w2id, alpha=0.0, eps=0):
        self.w2id = w2id
        self.eps = eps
        self.alpha = alpha
        self.unigrams = np.ones(len(self.w2id)) / len(self.w2id)
        self.bigrams = defaultdict(dd)
    
    def init(self):
        self.u_e = dy.inputTensor(self.unigrams)

    def p_next(self, sent):
        pw = sent#[s[-1] for s in sent]
        b_p = np.zeros((len(self.w2id), len(pw)))
        for i, w in enumerate(pw):
            for k, v in self.bigrams[w].items():
                b_p[k, i] = v
        return b_p

    def p_next_expr(self, sent):
        return dy.inputTensor(self.p_next(sent), batched=True)

    def fit(self, corpus):
        # Learn unigrams
        self.unigrams = np.zeros(len(self.w2id)) + self.eps
        for sent in corpus:
            for w in sent:
                self.unigrams[w] += 1
        self.unigrams /= self.unigrams.sum()
        # Learn bigrams
        for sent in corpus:
            for w, w_next in zip(sent[:-1], sent[1:]):
                self.bigrams[w][w_next] += 1
        for k, v in self.bigrams.items():
            s = sum(map(lambda x: x[1], v.items()))
            for w in v.keys():
                self.bigrams[k][w] /= s
        

    def save(self, filename):
        np.save(filename + '_unigrams', self.unigrams)
        with open(filename + '_bigrams', 'wb+') as f:
            pickle.dump(self.bigrams, f)

    def load(self, filename):
        self.unigrams = np.load(filename + '_unigrams')
        with open(filename + '_bigrams', 'rb') as f:
            self.bigrams = pickle.load(f)

def get_language_model(lm_type, w2id, test=False):
    if lm_type is None:
        return None
    if lm_type == 'uniform':
        return None
    elif lm_type == 'unigram':
        return lm.UnigramLanguageModel(w2id)
    elif lm_type == 'bigram':
        return lm.BigramLanguageModel(w2id)
    else:
        print('Unknown language model %s, using unigram language model' % lm_type)
        return lm.UnigramLanguageModel(w2id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a simple LM',
    )
    parser.add_argument('train_corpus', type=str, help='a path to the training corpus')
    parser.add_argument('out', type=str, help='Output file')
    parser.add_argument('--lm_type', '-lm', type=str, default=None,
                        help='Type of language model (for instance unigram or bigram)')
    parser.add_argument('--dic', '-d', type=str, default=None,
                        help='Path to a pickled dictionnary (maps words to ints). '
                        'If not provided, will learn from training data')
    parser.add_argument("--verbose", '-v',
                        help="increase output verbosity",
                        action="store_true")
    opt = parser.parse_args()
    # Load/create dic
    if opt.dic is None:
        dic = data.read_dic(opt.train_corpus)
    else:
        dic = data.load_dic(opt.dic)
    # Create LM object
    lm = get_language_model(opt.lm_type, dic)
    if lm is not None:
        # Load data
        train_data = data.read_corpus(opt.train_corpus, dic)
        # Train lm
        lm.fit(train_data)
        # Save lm
        lm.save(opt.out)
