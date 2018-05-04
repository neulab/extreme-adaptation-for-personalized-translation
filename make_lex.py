from __future__ import print_function, division

import options
import utils
import data
import lex

import sys

if __name__ == '__main__':
    opt = options.get_options()
    opt.lex_file = utils.exp_filename(opt, 'lex_file')
    lexicon = lex.Lexicon()
    lexicon.init(opt)
    trainingt_data = data.read_corpus(opt.train_trg, lexicon.w2idt)
    lexicon.trg_unigrams = lexicon.compute_unigrams(trainingt_data, 'trg')
    lex.save_lex(opt.lex_file, lexicon)

