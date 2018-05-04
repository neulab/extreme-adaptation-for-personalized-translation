from __future__ import print_function, division
from collections import defaultdict
import codecs
import sys


def load_data_file(filename):
    src, trg = [], []
    with open(filename, 'r') as f:
        for l in f:
            s, t = l.split(' ||| ')
            src.append(s.split())
            trg.append(t.split())
    return src, trg

def load_align_file(filename):
    aligns = []
    with open(filename, 'r') as f:
        for l in f:
            a = l.split()
            for i, pair in enumerate(a):
                pair = pair.split('-')
                a[i] = (int(pair[0]), int(pair[1]))
            aligns.append(a)
    return aligns

def compute_lexicon(src, trg, aligns):
    lex = defaultdict(lambda: defaultdict(lambda: 0.0))
    for s, t, a in zip(src, trg, aligns):
        for (i, j) in a:
            lex[s[i]][t[j]] = lex[s[i]][t[j]] + 1.0
    # Normalize
    for k in lex.keys():
        s = sum(map(lambda x: x[1], lex[k].items()))
        for w in lex[k].keys():
            lex[k][w] /= s
    return lex

def save_lexicon(filename, lex):
    with open(filename, 'w+') as f:
        for k, v in lex.items():
            for w, p in v.items():
                f.write('%s %s %f\n' % (k, w, p))

if __name__ == '__main__':
    assert len(sys.argv) == 4, 'Usage: python align_to_lexicon.py DATA_FILE ALIGN_FILE OUT_FILE'
    src, trg = load_data_file(sys.argv[1])
    aligns = load_align_file(sys.argv[2])
    lex = compute_lexicon(src, trg, aligns)
    save_lexicon(sys.argv[3], lex)
