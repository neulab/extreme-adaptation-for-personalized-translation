from __future__ import print_function, division

import sys

class Beam(object):
    def __init__(self, state, context, words, logprob, align=[0]):
        self.state = state
        self.words = words
        self.context = context
        self.logprob = logprob
        self.align = align
