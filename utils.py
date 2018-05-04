from __future__ import print_function, division

import time
import numpy as np

import sys


class Logger(object):
    """Simple logger object"""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def info(self, string):
        if self.verbose:
            print(string)
        sys.stdout.flush()


class Timer(object):
    """Simple timer"""

    def __init__(self, verbose=False):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def tick(self):
        elapsed = self.elapsed()
        self.restart()
        return elapsed


comparisons = {
    'lower': (lambda x, y: x < y),
    'greater': (lambda x, y: x > y)
}

worst = {
    'lower': np.inf,
    'greater': -np.inf
}


class EarlyStopping(object):
    """Handles early stopping"""

    def __init__(self, comparison, patience=0):

        self.comp = comparisons[comparison]
        self.best = worst[comparison]
        self.patience = patience
        self.deadline = 0

    def check(self, val):
        if self.comp(val, self.best):
            self.best = val
            self.deadline = 0
            return True
        else:
            self.deadline += 1
            return False

    def is_over(self):
        return self.deadline > self.patience


def exp_filename(opt, name):
    """Get a filename consistent with the experiment name and output dir"""
    return opt.output_dir + '/' + opt.exp_name + '_' + name


def exp_temp_filename(opt, name):
    """Get a filename consistent with the experiment name and temp dir"""
    return opt.temp_dir + '/' + opt.exp_name + '_' + name


def loadtxt(filename):
    """Simple text loading utility"""
    txt = []
    with open(filename, 'r') as f:
        for l in f:
            txt.append(l.strip())
    return txt


def savetxt(filename, txt):
    """Simple text saving utility"""
    with open(filename, 'w+') as f:
        for l in txt:
            print(l, file=f)
