from __future__ import print_function, division
import sys
from collections import defaultdict
import numpy as np
import dynet as dy
import pickle

def loadtxt(filename):
    txt = []
    with open(filename, 'r') as f:
        for l in f:
            txt.append(l.strip())
    return txt

def savetxt(filename, txt):
    with open(filename, 'w+') as f:
        for l in txt:
            print(l, file=f)

def txt2ngrams(X, ngrams):
    X_ngrams = []
    for sent in X:
        words = sent.lower().split()
        sent_ngrams = set()
        if len(words) == 0:
             X_ngrams.append([ngrams['.']])
             continue
        if words[0] in ngrams:
            sent_ngrams.add(ngrams[words[0]])
        for i in range(1, len(words)):
            if words[i] in ngrams:
                unigram = ngrams[words[i]]
                sent_ngrams.add(unigram)
            if " ".join(words[i:i+1]) in ngrams:
                bigram = ngrams[" ".join(words[i:i+1])]
                sent_ngrams.add(bigram)
        X_ngrams.append(list(sent_ngrams))
    X_ngrams = np.asarray(X_ngrams, dtype=list)
    return X_ngrams


train_file = sys.argv[1]
train_labels_file = sys.argv[2]
dev_file = sys.argv[3]
dev_labels_file = sys.argv[4]
dim = 128
batch_size = 32
num_epochs = 20

if sys.argv[5] == '--train':
    X = loadtxt(train_file)
    X_ngrams = []
    Y = loadtxt(train_labels_file)
    labels = list(set(Y))
    label_ids = {l:i for i,l in enumerate(labels)}
    Y_ids = np.asarray([label_ids[l] for l in Y], dtype=int)

    X_dev = loadtxt(dev_file)
    Y_dev = loadtxt(dev_labels_file)

    ngrams = defaultdict(lambda: len(ngrams))

    for sent, label in zip(X, Y):
        words = sent.lower().split()
        sent_ngrams = set([ngrams[words[0]]])
        for i in range(1, len(words)):
            unigram = ngrams[words[i]]
            bigram = ngrams[" ".join(words[i-1:i])]
            sent_ngrams.add(unigram)
            sent_ngrams.add(bigram)
        X_ngrams.append(list(sent_ngrams))
    X_ngrams = np.asarray(X_ngrams, dtype=list)

    ngrams = dict(ngrams)

    with open('ngrams.bin', 'wb+') as f:
        pickle.dump(ngrams, f)

    X_dev_ngrams = []
    for sent, label in zip(X_dev, Y_dev):
        words = sent.lower().split()
        if len(words)==0:
             X_dev_ngrams.append([ngrams['.']])
             continue
        sent_ngrams = set()
        if words[0] in ngrams:
            sent_ngrams.add(ngrams[words[0]])
        for i in range(1, len(words)):
            if words[i] in ngrams:
                unigram = ngrams[words[i]]
                sent_ngrams.add(unigram)
            if " ".join(words[i:i+1]) in ngrams:
                bigram = ngrams[" ".join(words[i:i+1])]
                sent_ngrams.add(bigram)
        X_dev_ngrams.append(list(sent_ngrams))
    X_dev_ngrams = np.asarray(X_dev_ngrams, dtype=list)


    Y_dev_ids = np.asarray([label_ids[l] for l in Y_dev], dtype=int)


    pc = dy.ParameterCollection()
    E = pc.add_lookup_parameters((len(ngrams), dim))
    W = pc.add_parameters((len(labels), dim))
    b = pc.add_parameters(len(labels))
    trainer = dy.AdamTrainer(pc)
    #trainer.set_clip_threshold(0)

    indices = np.arange(len(X), dtype=int)
    best_acc = 0
    for epoch in range(num_epochs):
        print('- Iteration %d' % (epoch+1), file=sys.stderr)
        np.random.shuffle(indices)
        for i in range(0, len(X) - 1, batch_size):
            dy.renew_cg()
            losses = []
            bsize = min(batch_size, len(X)-i)
            for sent, label in zip(X_ngrams[indices][i:i+bsize], Y_ids[indices][i:i+bsize]):
                bo_ngram = [dy.lookup(E, ngram) for ngram in sent]
                sent_vec = dy.average(bo_ngram)
                logits = dy.affine_transform([b.expr(), W.expr(), sent_vec])
                loss = dy.pickneglogsoftmax(logits, label)
                losses.append(loss)
            batch_loss = dy.average(losses)
            batch_loss.forward()
            batch_loss.backward()
            trainer.update()
            if (i//batch_size+1) % 1000 == 0:
                print('%.2f%% done, current loss: %.2f' % ((i+1) / (len(X)) * 100, batch_loss.value()), file=sys.stderr)
        
        accuracy = 0
        top5 = 0
        for sent, label in zip(X_dev_ngrams, Y_dev_ids):
            dy.renew_cg()
            bo_ngram = [dy.lookup(E, ngram) for ngram in sent]
            sent_vec = dy.average(bo_ngram)
            logits = sent_vec#dy.affine_transform([b.expr(), W.expr(), sent_vec])
            scores = logits.npvalue()
            ids = np.argsort(scores)
            accuracy += 1 if ids[-1] == label else 0
            top5 += 1 if label in ids[-5:] else 0
            
        print('Validation accuracy: %.3f%%' % (accuracy/len(X_dev)*100), file=sys.stderr)
        print('Validation top-5 accuracy: %.3f%%' % (top5/len(X_dev)*100), file=sys.stderr)

        if best_acc < accuracy:
            best_acc = accuracy
            pc.save('bo_ngrams.model')
elif sys.argv[5] == '--test':
    Y = loadtxt(train_labels_file)
    labels = list(set(Y))
    label_ids = {l:i for i,l in enumerate(labels)}
    Y_ids = np.asarray([label_ids[l] for l in Y], dtype=int)

    X_dev = loadtxt(dev_file)
    Y_dev = loadtxt(dev_labels_file)
    
    with open('ngrams.bin', 'rb') as f:
        ngrams = pickle.load(f)

    X_dev_ngrams = []
    for sent, label in zip(X_dev, Y_dev):
        words = sent.lower().split()
        sent_ngrams = set()
        if len(words)==0:
             X_dev_ngrams.append([ngrams['.']])
             continue
        if words[0] in ngrams:
            sent_ngrams.add(ngrams[words[0]])
        for i in range(1, len(words)):
            if words[i] in ngrams:
                unigram = ngrams[words[i]]
                sent_ngrams.add(unigram)
            if " ".join(words[i:i+1]) in ngrams:
                bigram = ngrams[" ".join(words[i:i+1])]
                sent_ngrams.add(bigram)
        X_dev_ngrams.append(list(sent_ngrams))
    X_dev_ngrams = np.asarray(X_dev_ngrams, dtype=list)


    Y_dev_ids = np.asarray([label_ids[l] for l in Y_dev], dtype=int)


    pc = dy.ParameterCollection()
    E = pc.add_lookup_parameters((len(ngrams), dim))
    W = pc.add_parameters((len(labels), dim))
    b = pc.add_parameters(len(labels))
    
    pc.populate('bo_ngrams.model')

    accuracy = 0
    top5 = 0
    top10 = 0
    for sent, label in zip(X_dev_ngrams, Y_dev_ids):
        dy.renew_cg()
        bo_ngram = [dy.lookup(E, ngram) for ngram in sent]
        sent_vec = dy.average(bo_ngram)
        logits = sent_vec#dy.affine_transform([b.expr(), W.expr(), sent_vec])
        scores = logits.npvalue()
        ids = np.argsort(scores)
        accuracy += 1 if ids[-1] == label else 0
        top5 += 1 if label in ids[-5:] else 0
        top10 += 1 if label in ids[-10:] else 0
    print('Evaluate on file %s' % dev_file)
    print('Test accuracy: %.3f%%' % (accuracy/len(X_dev)*100))
    print('Test top-5 accuracy: %.3f%%' % (top5/len(X_dev)*100))
    print('Test top-10 accuracy: %.3f%%' % (top10/len(X_dev)*100))
elif sys.argv[5] == '--qual':
    
    Y = loadtxt(train_labels_file)
    labels = list(set(Y))
    label_ids = {l:i for i,l in enumerate(labels)}
    Y_ids = np.asarray([label_ids[l] for l in Y], dtype=int)

    with open('ngrams.bin', 'rb') as f:
        ngrams = pickle.load(f)
    
    X_dev_1_text = loadtxt(train_file)
    X_dev_2_text = loadtxt(dev_file)
    src_txt = loadtxt('ted/en_fr/test.pretrain.en')
    trg_txt = loadtxt('ted/en_fr/test.pretrain.fr')

    X_dev_1 = txt2ngrams(X_dev_1_text, ngrams)
    X_dev_2 = txt2ngrams(X_dev_2_text, ngrams)
    Y_dev = loadtxt(dev_labels_file)
    Y_dev_ids = np.asarray([label_ids[l] for l in Y_dev], dtype=int)
    
    pc = dy.ParameterCollection()
    E = pc.add_lookup_parameters((len(ngrams), dim))
    W = pc.add_parameters((len(labels), dim))
    b = pc.add_parameters(len(labels))
    
    pc.populate('bo_ngrams.model')

    print('Comparing file %s to %s' % (dev_file, train_file))

    for i, (x1, x2, label) in enumerate(zip(X_dev_1, X_dev_2, Y_dev_ids)):
        dy.renew_cg()
        bo_ngram1 = [dy.lookup(E, ngram) for ngram in x1]
        sent_vec1 = dy.average(bo_ngram1)
        logits1 = dy.affine_transform([b.expr(), W.expr(), sent_vec1])
        scores1 = logits1.npvalue()
        pred1 = np.argmax(scores1)

        bo_ngram2 = [dy.lookup(E, ngram) for ngram in x2]
        sent_vec2 = dy.average(bo_ngram2)
        logits2 = dy.affine_transform([b.expr(), W.expr(), sent_vec2])
        scores2 = logits2.npvalue()
        pred2 = np.argmax(scores2)
        
        if pred1 != label and pred2 == label:
            print(labels[label])
            print('Source:\t %s' % src_txt[i])
            print('Fail:\t %s' % X_dev_1_text[i])
            print('Success:\t %s' % X_dev_2_text[i])
            print('Target:\t %s' % trg_txt[i])
            print('-'*80)

