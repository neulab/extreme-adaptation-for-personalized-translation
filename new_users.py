from __future__ import print_function, division

import options

import numpy as np
import dynet as dy

import data
import evaluation
import helpers
import utils

import sys
import os


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
    src = np.asarray(src)
    trg = np.asarray(trg)
    return src[ids[:-n_test]], src[ids[-n_test:]], trg[ids[:-n_test]], trg[ids[-n_test:]], ids

def evaluate_model(s2s, test_src, beam_size):
    s2s.set_test_mode()
    # Test on test split
    translations = []
    for x in test_src:
        y_hat = s2s.translate(x, 0, beam_size=beam_size)
        translations.append(y_hat)
    return translations

def adapt_user(s2s, trainer, train_src, train_trg, test_src,  opt):
    timer = utils.Timer()
    log = utils.Logger(opt.verbose)
    n_train = len(train_src)
    n_tokens = (sum(map(len, train_trg)) - len(train_trg))
    # Train for n_iter
    timer.restart()
    best_ppl = np.inf
    for epoch in range(opt.num_epochs):
        timer.tick()
        dy.renew_cg()
        losses = []
        # Add losses for all samples
        for x, y in zip(train_src, train_trg):
            losses.append(s2s.calculate_user_loss([x], [y], [0], update_mode=opt.update_mode))
        loss = dy.average(losses)
        # Backward + update
        loss.backward()
        trainer.update()
        # Print loss etc...
        train_loss = loss.value() / n_tokens
        train_ppl = np.exp(train_loss)
        trainer.status()
        elapsed = timer.tick()
        log.info(" Training_loss=%f, ppl=%f, time=%f s, tok/s=%.1f" %
             (train_loss, train_ppl, elapsed, n_tokens / elapsed))
        if train_ppl < best_ppl:
            best_ppl = train_ppl
            translations = evaluate_model(s2s, test_src, opt.beam_size)
        else:
            log.info("Early stopping after %d iterations" % (epoch+1))
            break
    return translations

    
def optimized_adapt():
    # Precompute scores if we're only training the biases
    train_scores = []
    for x, y in zip(train_src, train_trg):
        sent_scores = s2s.precompute_scores([x], [y], [0])
        train_scores.append([score.npvalue() for score in sent_scores])
    #print('[%s]' % ', '.join(str(len(s)) for s in train_scores))
    # Train
    timer.restart()
    best_ppl = np.inf
    for epoch in range(opt.num_epochs):
        timer.tick()
        dy.renew_cg()
        losses = []
        # Add losses for all samples
        for x, y in zip(train_scores, train_trg):
            losses.append(s2s.calculate_user_bias_loss([x], [y], [0], update_mode=opt.update_mode))
        loss = dy.average(losses)
        # Backward + update
        loss.backward()
        trainer.update()
        # Print loss etc...
        train_loss = loss.value() / n_tokens
        train_ppl = np.exp(train_loss)
        elapsed = timer.tick()
        if (epoch + 1) % opt.check_train_error_every == 0:
            trainer.status()
            log.info(" Training_loss=%f, ppl=%f, time=%f s, tok/s=%.1f" %
                 (train_loss, train_ppl, elapsed, n_tokens / elapsed))
        if train_ppl < best_ppl and best_ppl - train_ppl >= 1e-3:
            best_ppl = train_ppl
            translations = evaluate_model(s2s, test_src, opt.beam_size)
        else:
            log.info("Early stopping after %d iterations" % (epoch+1))
            break
        # Record metrics


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
    if opt.update_mode == 'mixture_weights' and not opt.user_recognizer == 'fact_voc':
        log.info('Updating only the mixture weights doesn\'t make sense here')
        exit()
    s2s.lm = lexicon.trg_unigrams
    #    s2s.freeze_parameters()
    # Trainer
    trainer = helpers.get_trainer(opt, s2s)
    # print config
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(lexicon.w2ids),
                             trg_dict_size=len(lexicon.w2idt))
    # This will store translations and gold sentences
    base_translations = []
    adapt_translations = []
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
        # Reset model
        s2s.load()
        s2s.reset_usr_vec()
        # Translate with baseline model
        base_translations.extend(evaluate_model(s2s, test_src, opt.beam_size))
        # Start loop
        n_train = opt.max_n_train
        adapt_translations.extend(adapt_user(s2s, trainer, train_src[:n_train], train_trg[:n_train], test_src, opt))

    # Temp files
    temp_gold = utils.exp_temp_filename(opt, 'gold.txt')
    temp_base = utils.exp_temp_filename(opt, '%s_base.txt' % opt.update_mode)
    temp_adapt = utils.exp_temp_filename(opt, '%s_adapt.txt' % opt.update_mode)
    utils.savetxt(temp_gold, gold)
    utils.savetxt(temp_base, base_translations)
    utils.savetxt(temp_adapt, adapt_translations)
    # Evaluate base translations
    bleu, details = evaluation.bleu_score(temp_gold, temp_base)
    log.info('Base BLEU score: %.2f' % bleu)
    # Evaluate base translations
    bleu, details = evaluation.bleu_score(temp_gold, temp_adapt)
    log.info('Adaptation BLEU score: %.2f' % bleu)
    # Compare both
    temp_bootstrap_gold = utils.exp_temp_filename(opt, 'bootstrap_gold.txt')
    temp_bootstrap_base = utils.exp_temp_filename(opt, 'bootstrap_base.txt')
    temp_bootstrap_adapt = utils.exp_temp_filename(opt, 'bootstrap_adapt.txt')
    bleus = evaluation.paired_bootstrap_resampling(temp_gold, temp_base, temp_adapt,
            opt.bootstrap_num_samples,
            opt.bootstrap_sample_size,
            temp_bootstrap_gold,
            temp_bootstrap_base,
            temp_bootstrap_adapt)
    evaluation.print_paired_stats(bleus)
    os.remove(temp_bootstrap_gold)
    os.remove(temp_bootstrap_base)
    os.remove(temp_bootstrap_adapt)
    # Results

if __name__ == "__main__":
    opt = options.get_options()
    eval_user_adaptation(opt)
