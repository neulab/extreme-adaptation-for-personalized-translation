from __future__ import print_function, division

import numpy as np
import dynet as dy

import sys


class Attention(object):
    """docstring for Attention"""

    def __init__(self, pc):
        self.pc = pc.add_subcollection('att')

    def init(self, test=True, update=True):
        pass

    def __call__(self, H, h, test=True):
        raise NotImplemented()


class EmptyAttention(Attention):
    """docstring for EmptyAttention"""

    def __init__(self, pc):
        super(EmptyAttention, self).__init__(pc)

    def __call__(self, H, h, test=True):
        return 0, 0


class MLPAttention(Attention):
    """docstring for MLPAttention"""

    def __init__(self, di, dh, da, pc):
        super(MLPAttention, self).__init__(pc)
        self.di, self.dh, self.da = di, dh, da
        # Parameters
        self.Va_p = self.pc.add_parameters((self.da), name='Va')
        self.Wa_p = self.pc.add_parameters((self.da, self.di), name='Wa')
        self.Wha_p = self.pc.add_parameters((self.da, self.dh), name='Wha')

    def init(self, test=True, update=True):
        self.Va = self.Va_p.expr(update)
        self.Wa = self.Wa_p.expr(update)
        self.Wha = self.Wha_p.expr(update)
    
    def __call__(self, H, h, test=True):
        d = dy.tanh(dy.colwise_add(self.Wa * H, self.Wha * h))
        scores = dy.transpose(d) * self.Va
        weights = dy.softmax(scores)
        context = H * weights
        return context, weights

class MultiHeadAttention(Attention):
    """docstring for MultiHeadAttention"""

    def __init__(self, nh, di, dh, da, pc):
        super(MultiHeadAttention, self).__init__(pc)
        self.nh, self.di, self.dh, self.da = nh, di, dh, da
        self.d = self.da // self.nh
        self.scale = 1 / np.sqrt(self.d)
        assert self.da % self.nh == 0, 'The attention dimension must be a multiple of the number of heads'
        self.KW_p = self.pc.add_parameters((self.da, self.di))
        self.QW_p = self.pc.add_parameters((self.da, self.dh))
        self.VW_p = self.pc.add_parameters((self.da, self.di))
        self.OW_p = self.pc.add_parameters((self.di, self.da))

    def init(self, test=True, update=True):
        self.KW = self.KW_p.expr(update)
        self.QW = self.QW_p.expr(update)
        self.VW = self.VW_p.expr(update)
        self.OW = self.OW_p.expr(update)


    def __call__(self, H, h, test=True):
        K = self.KW * H
        Q = self.QW * h
        V = self.VW * H
        o = []
        w = []
        for i in range(self.nh):
            K_i = K[i*self.d:(i+1)*self.d]
            Q_i = Q[i*self.d:(i+1)*self.d]
            V_i = V[i*self.d:(i+1)*self.d]
            w_i = (dy.transpose(K_i) * Q_i) * self.scale
            o_i = V_i * w_i
            o.append(o_i)
            w.append(w_i)
        c = self.OW * dy.concatenate(o)
        return c, dy.concatenate(w)

def get_attention(attention, di, dh, da, pc):
    if attention == 'empty':
        return EmptyAttention(pc)
    elif attention == 'mlp':
        return MLPAttention(di, dh, da, pc)
    elif attention == 'multi_head':
        return MultiHeadAttention(8 ,di, dh, da, pc)
    else:
        print('Unknown attention type "%s", using mlp attention' % attention)
        return MLPAttention(di, dh, da, pc)
