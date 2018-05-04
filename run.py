from __future__ import print_function, division

import options

import numpy as np

import data
import evaluation
import helpers
import utils

import sys


def train(opt):
    log = utils.Logger(opt.verbose)
    timer = utils.Timer()
    # Load data =========================================================
    log.info('Reading corpora')
    # Read vocabs
    lexicon = helpers.get_lexicon(opt)
    # Read training
    trainings_data = data.read_corpus(opt.train_src, lexicon.w2ids)
    trainingt_data = data.read_corpus(opt.train_trg, lexicon.w2idt)
    training_usr_data = data.read_talk(opt.train_usr, lexicon.usr2id)
    # Read validation
    valids_data = data.read_corpus(opt.valid_src, lexicon.w2ids)
    validt_data = data.read_corpus(opt.valid_trg, lexicon.w2idt)
    valid_usr_data = data.read_talk(opt.valid_usr, lexicon.usr2id)
    # Validation output
    if not opt.valid_out:
        opt.valid_out = utils.exp_filename(opt, 'valid.out')
    # Get target language model
    lang_model = helpers.get_language_model(opt, trainingt_data, lexicon.w2idt)
    # Create model ======================================================
    log.info('Creating model')
    s2s = helpers.build_model(opt, lexicon, lang_model)
    # Trainer ==========================================================
    trainer = helpers.get_trainer(opt, s2s)
    log.info('Using ' + opt.trainer + ' optimizer')
    # Print configuration ===============================================
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(lexicon.w2ids),
                             trg_dict_size=len(lexicon.w2idt))
    # Creat batch loaders ===============================================
    log.info('Creating batch loaders')
    trainbatchloader = data.BatchLoader(
        trainings_data, trainingt_data, training_usr_data, opt.batch_size)
    devbatchloader = data.BatchLoader(valids_data, validt_data, valid_usr_data, opt.dev_batch_size)
    # Start training ====================================================
    log.info('starting training')
    timer.restart()
    train_loss = 0
    train_user_nll = 0
    processed = 0
    best_bleu = -1
    best_ppl = np.inf
    deadline = 0
    i = 0
    for epoch in range(opt.num_epochs):
        for batch in trainbatchloader:
            s2s.set_train_mode()
            processed += sum(map(len, batch.trg))
            bsize = len(batch.trg)
            # Compute loss
            if opt.user_training:
                decode_nll, user_nll = s2s.calculate_user_loss(batch.src, batch.trg, batch.usr)
                nll = decode_nll + user_nll
            else:
                nll = decode_nll = s2s.calculate_loss(batch.src, batch.trg, batch.usr)
            # Backward pass and parameter update
            nll.backward()
            trainer.update()
            train_loss += decode_nll.scalar_value() * bsize
            if opt.user_training:
                train_user_nll = user_nll.scalar_value() * bsize
            if (i + 1) % opt.check_train_error_every == 0:
                # Check average training error from time to time
                logloss = train_loss / processed
                ppl = np.exp(logloss)
                trainer.status()
                if opt.user_training:
                    log.info(" Training_loss=%f, user_nll=%.2f, ppl=%f, time=%f s, tokens processed=%d" %
                         (logloss, train_user_nll / processed, ppl, timer.tick(), processed))
                else:
                    log.info(" Training_loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                         (logloss, ppl, timer.tick(), processed))
                train_loss = 0
                train_user_nll = 0
                processed = 0
            i = i + 1
        # Check generalization error on the validation set from time to time
        s2s.set_test_mode()
        dev_loss = 0
        dev_processed = 0
        timer.restart()
        for dev_batch in devbatchloader:
            dev_processed += sum(map(len, dev_batch.trg))
            bsize = len(dev_batch.trg)
            loss = s2s.calculate_loss(dev_batch.src, dev_batch.trg,
                                      dev_batch.usr, test=True)
            dev_loss += loss.scalar_value() * bsize
        dev_logloss = dev_loss / dev_processed
        dev_ppl = np.exp(dev_logloss)
        log.info("[epoch %d] Dev loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                 (epoch, dev_logloss, dev_ppl, timer.tick(), dev_processed))
        # Early stopping : save the latest best model
        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            log.info('Best perplexity up to date (%.2f), saving model to %s' % (dev_ppl, s2s.model_file))
            s2s.save()
            deadline = 0
        else:
            deadline += 1
            # Reload previous checkpoint
            s2s.load()
            # Restart trainer
            trainer.restart()
            trainer.learning_rate *= opt.learning_rate_decay
        if opt.patience > 0 and deadline > opt.patience:
            log.info('No improvement since %d epochs, early stopping '
                     'with best validation BLEU score: %.3f' % (deadline, best_bleu))
            exit()

        # Check BLEU score on the validation set from time to time
        s2s.set_test_mode()
        log.info('Start translating validation set, buckle up!')
        timer.restart()
        with open(opt.valid_src, 'r') as f:
            translations = []
            for l, t in zip(f, valid_usr_data):
                y_hat = s2s.translate(l.split(), t, beam_size=opt.beam_size)
                translations.append(y_hat)
        np.savetxt(opt.valid_out, translations, fmt='%s')
        bleu, details = evaluation.bleu_score(opt.valid_trg, opt.valid_out)
        log.info('Finished translating validation set %.2f elapsed.' % timer.tick())
        log.info(details)


def test(opt):
    log = utils.Logger(opt.verbose)
    timer = utils.Timer()
    # Load data =========================================================
    log.info('Reading corpora')
    # Read vocabs
    lexicon = helpers.get_lexicon(opt)
    # Read test
    test_usr_data = data.read_talk(opt.test_usr, lexicon.usr2id)
    # Test output
    if not opt.test_out:
        opt.test_out = utils.exp_filename(opt, 'test.out')
    # Get target language model
    lang_model = helpers.get_language_model(opt, None, lexicon.w2idt, test=True)
    # Create model ======================================================
    log.info('Creating model')
    s2s = helpers.build_model(opt, lexicon, lang_model, test=True)
    # Print configuration ===============================================
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(lexicon.w2ids),
                             trg_dict_size=len(lexicon.w2idt))
    # Start testing =====================================================
    log.info('Start running on test set, buckle up!')
    timer.restart()
    translations = []
    s2s.set_test_mode()
    with open(opt.test_src, 'r') as f:
        for i, (l, t) in enumerate(zip(f, test_usr_data)):
            y = s2s.translate(l.split(), t, beam_size=opt.beam_size)
            translations.append(y)
    np.savetxt(opt.test_out, translations, fmt='%s')
    BLEU, details = evaluation.bleu_score(opt.test_trg, opt.test_out)
    log.info('Finished running on test set %.2f elapsed.' % timer.tick())
    log.info(details)


def interactive(opt):
    # Load data =========================================================
    if opt.verbose:
        print('Reading corpora')
    # Read vocabs
    widss, ids2ws, widst, ids2wt, tids, ids2t = get_dictionaries(opt, True)
    # Create model ======================================================
    if opt.verbose:
        print('Creating model')
        sys.stdout.flush()
    s2s = build_model(opt, widss, widst, tids)
    if s2s.model_file is None:
        s2s.model_file = opt.output_dir + '/' + opt.exp_name + '_model.txt'
    print('loading from ' + s2s.model_file)
    s2s.load()
    # Print configuration ===============================================
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(widss), trg_dict_size=len(widst))
        sys.stdout.flush()
    return s2s

if __name__ == '__main__':
    # Retrieve options ==================================================
    opt = options.get_options()
    if opt.train:
        train(opt)
    elif opt.test:
        test(opt)
