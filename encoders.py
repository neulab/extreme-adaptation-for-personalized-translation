from __future__ import print_function, division

import dynet as dy
import numpy as np

import sys


class Encoder(object):
    """Base Encoder class"""

    def __init__(self, pc):
        self.pc = pc.add_subcollection('enc')
        self.dim = 0

    def init(self, x, usr, test=True, update=True):
        pass

    def __call__(self, x, test=True):
        raise NotImplemented()


class EmptyEncoder(Encoder):
    """docstring for EmptyEncoder"""

    def __init__(self, pc):
        super(EmptyEncoder, self).__init__(pc)

    def __call__(self, x, test=True, update=True):
        return 0


class LSTMEncoder(Encoder):
    """docstring for LSTMEncoder"""

    def __init__(self, nl, di, dh, vs, pc, dr=0.0, pre_embs=None):
        super(LSTMEncoder, self).__init__(pc)
        # Store hyperparameters
        self.nl, self.di, self.dh = nl, di, dh
        self.dr = dr
        self.vs = vs
        self.dim += dh
        # LSTM Encoder
        self.lstm = dy.VanillaLSTMBuilder(self.nl, self.di, self.dh, self.pc)
        # Embedding matrix
        self.E = self.pc.add_lookup_parameters((self.vs, self.di), name='E')
        if pre_embs is not None:
            self.E.init_from_array(pre_embs)

    def init(self, x, usr, test=True, update=True):
        bs = len(x[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Add encoder to computation graph
        self.es = self.lstm.initial_state(update=update)
        if not test:
            self.lstm.set_dropout_masks(bs)

    def __call__(self, x, test=True, update=True):
        wembs = [dy.lookup_batch(self.E, iw, update=update) for iw in x]
        # Encode sentence
        encoded_states = self.es.transduce(wembs)
        # Create encoding matrix
        H = dy.concatenate_cols(encoded_states)
        return H

class UserLSTMEncoder(LSTMEncoder):
    def __init__(self, nl, di, dh, du, vs, pc, dr=0.0, pre_embs=None):
        super(UserLSTMEncoder, self).__init__(nl, di, dh, vs, pc, dr, pre_embs)
        self.du = du
        self.Th_p = self.pc.add_parameters((dh, du), init=dy.UniformInitializer(1/np.sqrt(dh)), name='Th')

    def init(self, x, usr, test=True, update=True):
        bs = len(x[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Add encoder to computation graph
        self.Th = self.Th_p.expr(update)
        init_state = self.Th * usr
        init_state = [init_state, dy.zeroes((self.dh,), batch_size=bs)]
        self.es = self.lstm.initial_state(init_state, update=update)
        if not test:
            self.lstm.set_dropout_masks(bs)
        
class BiLSTMEncoder(LSTMEncoder):
    """docstring for BiLSTMEncoder"""

    def __init__(self, nl, di, dh, vs, pc, dr=0.0, pre_embs=None):
        super(BiLSTMEncoder, self).__init__(nl, di, dh, vs, pc, dr, pre_embs)
        self.dim += dh
        # Backward encoder
        self.rev_lstm = dy.VanillaLSTMBuilder(self.nl, self.di, self.dh, self.pc)

    def init(self, x, usr, test=True, update=True):
        super(BiLSTMEncoder, self).init(x, usr, test, update)
        bs = len(x[0])
        if not test:
            self.rev_lstm.set_dropout(self.dr)
        else:
            self.rev_lstm.disable_dropout()
        # Add encoder to computation graph
        self.res = self.rev_lstm.initial_state(update=update)

        if not test:
            self.rev_lstm.set_dropout_masks(bs)

    def __call__(self, x, test=True, update=True):
        # Embed words
        wembs = [dy.lookup_batch(self.E, iw, update) for iw in x]
        # Encode sentence
        encoded_states = self.es.transduce(wembs)
        rev_encoded_states = self.res.transduce(wembs[::-1])[::-1]
        # Create encoding matrix
        H_fwd = dy.concatenate_cols(encoded_states)
        H_bwd = dy.concatenate_cols(rev_encoded_states)
        H = dy.concatenate([H_fwd, H_bwd])

        return H

class BiUserLSTMEncoder(UserLSTMEncoder):
    """docstring for BiLSTMEncoder"""

    def __init__(self, nl, di, dh, du, vs, pc, dr=0.0, pre_embs=None):
        super(BiUserLSTMEncoder, self).__init__(nl, di, dh, du, vs, pc, dr, pre_embs)
        self.dim += dh
        # Backward encoder
        self.rev_lstm = dy.VanillaLSTMBuilder(self.nl, self.di, self.dh, self.pc)
        
        self.rev_Th_p = self.pc.add_parameters((dh, du), init=dy.UniformInitializer(1/np.sqrt(dh)), name='revTh')

    def init(self, x, usr, test=True, update=True):
        super(BiUserLSTMEncoder, self).init(x, usr, test, update)
        bs = len(x[0])
        if not test:
            self.rev_lstm.set_dropout(self.dr)
        else:
            self.rev_lstm.disable_dropout()
        # Add encoder to computation graph
        self.rev_Th = self.rev_Th_p.expr(update)
        init_state = self.rev_Th * usr
        init_state = [init_state, dy.zeroes((self.dh,), batch_size=bs)]
        self.res = self.rev_lstm.initial_state(init_state, update=update)

        if not test:
            self.rev_lstm.set_dropout_masks(bs)

    def __call__(self, x, test=True, update=True):
        # Embed words
        wembs = [dy.lookup_batch(self.E, iw) for iw in x]
        # Encode sentence
        encoded_states = self.es.transduce(wembs)
        rev_encoded_states = self.res.transduce(wembs[::-1])[::-1]
        # Create encoding matrix
        H_fwd = dy.concatenate_cols(encoded_states)
        H_bwd = dy.concatenate_cols(rev_encoded_states)
        H = dy.concatenate([H_fwd, H_bwd])

        return H

def get_encoder(encoder, nl, di, dh, du, vs, pc, dr=0.0, pre_embs=None):
    if encoder == 'empty':
        return EmptyEncoder(pc)
    elif encoder == 'lstm':
        return LSTMEncoder(nl, di, dh, vs, pc, dr=dr, pre_embs=pre_embs)
    elif encoder == 'bilstm':
        return BiLSTMEncoder(nl, di, dh, vs, pc, dr=dr, pre_embs=pre_embs)
    elif encoder == 'user_lstm':
        return UserLSTMEncoder(nl, di, dh, du,vs, pc, dr=dr, pre_embs=pre_embs)
    elif encoder == 'user_bilstm':
        return BiUserLSTMEncoder(nl, di, dh, du, vs, pc, dr=dr, pre_embs=pre_embs)
    else:
        print('Unknown encoder type "%s", using bilstm encoder' % encoder)
        return BiLSTMEncoder(nl, di, dh, vs, pc, dr=dr, pre_embs=pre_embs)
