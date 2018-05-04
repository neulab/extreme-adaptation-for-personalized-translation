from __future__ import print_function, division

import time

import numpy as np
import dynet as dy

import data
import seq2seq
import lm
import lex
import utils

import sys



def load_pretrained_wembs(opt, lexicon):
    if opt.pretrained_wembs is not None:
        print('Using pretrained word embeddings from %s' % opt.pretrained_wembs)
        wv = data.load_word_vectors(opt.pretrained_wembs, lexicon.w2idt)
        d = wv.shape[1]
        pretrained_wembs = np.zeros((len(lexicon.w2idt), opt.emb_dim))
        pretrained_wembs[:, (opt.emb_dim - d):] = wv
    else:
        pretrained_wembs = None
    return pretrained_wembs

def load_pretrained_user(opt, lexicon):
    if opt.pretrained_user is not None:
        print('Using pretrained user embeddings from %s' % opt.pretrained_user)
        uv = np.load(opt.pretrained_user)
        d = uv.shape[1]
        pretrained_user = np.zeros((len(lexicon.usr2id), opt.usr_dim))
        pretrained_user[:, (opt.usr_dim - d):] = uv
    else:
        pretrained_user = None
    return pretrained_user

def build_model(opt, lexicon, lang_model, test=False):
    s2s = seq2seq.Seq2SeqModel(opt,
                               lexicon,
                               lang_model=lang_model,
                               pretrained_wembs=load_pretrained_wembs(opt, lexicon))
    s2s.set_usr(opt.user_recognizer, pretrained_user=load_pretrained_user(opt, lexicon))
    if test or opt.pretrained:
        if s2s.model_file is None:
            s2s.model_file = utils.exp_filename(opt, 'model')
        print('loading pretrained model from %s' % s2s.model_file)
        s2s.load()
    else:
        if s2s.model_file is not None:
            s2s.load()
        s2s.model_file = utils.exp_filename(opt, 'model')

    #if opt.user_training:

    return s2s


def get_lexicon(opt):
    load = not opt.train
    if opt.lex_file is None:
        opt.lex_file = utils.exp_filename(opt, 'lex_file')
    else:
        load = True
    if opt.train and not load:
        lexicon = lex.Lexicon()
        lexicon.init(opt)
        lex.save_lex(opt.lex_file, lexicon)
    else:
        if opt.lex_file is None:
            opt.lex_file = utils.exp_filename(opt, 'lex_file')
        print('Loading lexicon from file: %s' % opt.lex_file)
        lexicon = lex.load_lex(opt.lex_file)
    return lexicon


def get_language_model(opt, train_data, w2id, test=False):
    if opt.language_model is None:
        return None
    if opt.language_model == 'uniform':
        return None
    elif opt.language_model == 'unigram':
        lang_model = lm.UnigramLanguageModel(w2id)
    elif opt.language_model == 'bigram':
        lang_model = lm.BigramLanguageModel(w2id)
    else:
        print('Unknown language model %s, using unigram language model' % opt.language_model)
        lang_model = lm.UnigramLanguageModel(w2id)

    if opt.lm_file is not None or test:
        if opt.lm_file is None:
            opt.lm_file = utils.exp_filename(opt, 'lm')
        lang_model.load(opt.lm_file)
    else:
        print('training lm')
        lang_model.fit(train_data)
        opt.lm_file = utils.exp_filename(opt, 'lm')
        lang_model.save(opt.lm_file)
    return lang_model


def get_trainer(opt, s2s):
    if opt.trainer == 'sgd':
        trainer = dy.SimpleSGDTrainer(s2s.pc,
                                      learning_rate=opt.learning_rate)
    elif opt.trainer == 'clr':
        trainer = dy.CyclicalSGDTrainer(s2s.pc,
                                        learning_rate_min=opt.learning_rate / 10.0,
                                        learning_rate_max=opt.learning_rate)
    elif opt.trainer == 'momentum':
        trainer = dy.MomentumSGDTrainer(s2s.pc,
                                        learning_rate=opt.learning_rate)
    elif opt.trainer == 'rmsprop':
        trainer = dy.RMSPropTrainer(s2s.pc,
                                    learning_rate=opt.learning_rate)
    elif opt.trainer == 'adam':
        trainer = dy.AdamTrainer(s2s.pc,
                                 opt.learning_rate)
    else:
        print('Trainer name invalid or not provided, using SGD', file=sys.stderr)
        trainer = dy.SimpleSGDTrainer(s2s.pc,
                                      learning_rate=opt.learning_rate)

    trainer.set_clip_threshold(opt.gradient_clip)

    return trainer
