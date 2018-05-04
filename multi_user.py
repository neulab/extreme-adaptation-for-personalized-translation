from __future__ import print_function, division

import options

import numpy as np
import dynet as dy

import data
import evaluation
import helpers
import utils

import sys
import os.path


def load_user_filepairs(file_list):
    src_files, trg_files = [], []
    with open(file_list, 'r') as f:
        for l in f:
            src_file, trg_file = l.strip().split()
            src_files.append(src_file)
            trg_files.append(trg_file)
    return zip(src_files, trg_files)


def split_user_data(src, trg, n_test=2):
    ids = np.arange(len(src), dtype=int)
    np.random.shuffle(ids)
    return src[:-n_test], src[-n_test:], trg[:-n_test], trg[-n_test:], ids

def adapt(s2s, trainer, X, Y, n_epochs, check_train_error_every):
    timer = utils.Timer()
    log = utils.Logger(True)
    n_train = len(X)
    n_tokens = (sum(map(len, Y)) - len(Y))
    s2s.set_train_mode()
    s2s.reset_usr_vec()
    # Train for n_iter
    for epoch in range(n_epochs):
        dy.renew_cg()
        loss = dy.zeros((1,))
        timer.restart()
        # Add losses for all samples
        for x, y in zip(X, Y):
            loss += s2s.calculate_user_loss([x], [y])
        # Backward + update
        loss.backward()
        trainer.update()
        # Record metrics
        if n_train > 0 and epoch % check_train_error_every == 0:
            train_loss = loss.value() / n_tokens
            train_ppl = np.exp(train_loss)
            trainer.status()
            elapsed = timer.tick()
            log.info(" Training_loss=%f, ppl=%f, time=%f s, tok/s=%.1f" %
                 (train_loss, train_ppl, elapsed, n_tokens / elapsed))

def eval_user_adaptation(opt):
    log = utils.Logger(opt.verbose)
    timer = utils.Timer()
    # Read vocabs
    lexicon = helpers.get_lexicon(opt)
    # Read data
    filepairs = load_user_filepairs(opt.usr_file_list)
    # Get target language model
    lang_model = None
    # Load model
    s2s = helpers.build_model(opt, lexicon, lang_model, test=True)
    #if not opt.full_training:
    #    s2s.freeze_parameters()
    # Trainer
    trainer = helpers.get_trainer(opt, s2s)
    # print config
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(lexicon.w2ids),
                             trg_dict_size=len(lexicon.w2idt))
    # This will store translations and gold sentences
    translations = dict([(i, []) for i in range(opt.min_n_train, opt.max_n_train)])
    gold = []
    # Run training
    for usr_id, (src_file, trg_file) in enumerate(filepairs):
        log.info('Evaluating on files %s' % os.path.basename(src_file).split()[0])
        # Load file pair
        src_data = data.read_corpus(src_file, lexicon.w2ids, raw=True)
        trg_data = data.read_corpus(trg_file, lexicon.w2idt, raw=True)
        # split train/test
        train_src, test_src, train_trg, test_trg, order = split_user_data(
            src_data, trg_data, n_test=opt.n_test)
        # Convert train data to indices
        train_src = lexicon.sents_to_ids(train_src)
        train_trg = lexicon.sents_to_ids(train_trg, trg=True)
        # Save test data
        for s in test_trg:
            gold.append(' '.join(s))
        # Start loop
        for n_train in range(opt.min_n_train, opt.max_n_train):
            log.info('Training on %d sentence pairs' % n_train)
            # Train on n_train first sentences
            X, Y = train_src[:n_train], train_trg[:n_train]
            temp_out = utils.exp_temp_filename(opt, str(n_train) + 'out.txt')
            if opt.full_training:
                s2s.load()
            if opt.log_unigram_bias:
                if opt.use_trg_unigrams:
                    unigrams = lexicon.compute_unigrams(Y, lang='trg')
                else:
                    unigrams = lexicon.estimate_unigrams(X)
                log_unigrams = np.log(unigrams + opt.log_unigrams_eps)
                s2s.reset_usr_vec(log_unigrams)
            elif n_train > 0:
                adapt(s2s, trainer, X, Y, opt.num_epochs, opt.check_train_error_every)
            log.info('Translating test file')
            s2s.set_test_mode()
            # Test on test split
            for x in test_src:
                y_hat = s2s.translate(x, 0, beam_size=opt.beam_size)
                translations[n_train].append(y_hat)

    # Temp files
    temp_gold = utils.exp_temp_filename(opt, 'gold.txt')
    np.savetxt(temp_gold, gold, fmt='%s')
    # Results
    test_bleus = np.zeros(opt.max_n_train - opt.min_n_train)
    for n_train in range(opt.min_n_train, opt.max_n_train):
        log.info('Evaluation for %d sentence pairs' % n_train)
        temp_out = utils.exp_temp_filename(opt, str(n_train) + 'out.txt')
        temp_bootstrap_out = utils.exp_temp_filename(opt, str(n_train) + '_bootstrap_out.txt')
        temp_bootstrap_ref = utils.exp_temp_filename(opt, str(n_train) + '_bootstrap_ref.txt')
        np.savetxt(temp_out, translations[n_train], fmt='%s')
        bleu, details = evaluation.bleu_score(temp_gold, temp_out)
        log.info('BLEU score: %.2f' % bleu)
        bleus = evaluation.bootstrap_resampling(temp_gold, temp_out, opt.bootstrap_num_samples, opt.bootstrap_sample_size, temp_bootstrap_ref, temp_bootstrap_out)
        evaluation.print_stats(bleus)
        test_bleus[n_train - opt.min_n_train] = bleu
    np.savetxt(utils.exp_filename(opt, 'bleu_scores.txt'), test_bleus, fmt='%.3f')


if __name__ == "__main__":
    opt = options.get_options()
    eval_user_adaptation(opt)
