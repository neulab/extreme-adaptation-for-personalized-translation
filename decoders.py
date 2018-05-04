from __future__ import print_function, division

import numpy as np
import dynet as dy

import sys


class Decoder(object):
    """Base Encoder class"""

    def __init__(self, pc):
        self.pc = pc.add_subcollection('dec')

    def init(self, H, y, usr, test=True, update=True):
        pass

    def next(self, w, c, test=True, state=None):
        raise NotImplemented()

    def s(self, h, c, e, test=True):
        raise NotImplemented()


class LSTMLMDecoder(Decoder):
    """docstring for EmptyEncoder"""

    def __init__(self, nl, di, dh, vt, pc, pre_embs=None, dr=0.0, wdr=0.0):
        super(LSTMLMDecoder, self).__init__(pc)

        # Store hyperparameters
        self.nl, self.di, self.dh = nl, di, dh
        self.dr, self.wdr = dr, wdr
        self.vt = vt
        # LSTM Encoder
        self.lstm = dy.VanillaLSTMBuilder(self.nl, self.di, self.dh, self.pc)
        # Output layer
        self.Wo_p = self.pc.add_parameters((self.di, self.dh + self.di), name='Wo')
        self.bo_p = self.pc.add_parameters((self.di,), name='bo')
        # Embedding matrix
        self.E_p = self.pc.add_parameters((self.vt, self.di), name='E')
        if pre_embs is not None:
            self.E.set_value(pre_embs)

    def init(self, H, y, usr, test=True, update=True):
        bs = len(y[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Add encoder to computation graph
        self.ds = self.lstm.initial_state(update=update)
        if not test:
            self.lstm.set_dropout_masks(bs)

        self.Wo = self.Wo_p.expr(update)
        self.bo = self.bo_p.expr(update)

        self.E = self.E_p.expr(update)

    def next(self, w, c, test=True, state=None):
        e = dy.pick_batch(self.E, w)
        if not test:
            e = dy.dropout_dim(e, 0, self.wdr)
        # Run LSTM
        if state is None:
            self.ds = self.ds.add_input(e)
            next_state = self.ds
        else:
            next_state = state.add_input(e)
        h = next_state.output()
        return h, e, next_state

    def s(self, h, c, e, test=True):
        output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([h, e])])
        if not test:
            output = dy.dropout(output, self.dr)
        # Score
        s = self.E * output
        return s


class LSTMDecoder(Decoder):
    """docstring for LSTMDecoder"""

    def __init__(self, nl, di, de, dh, vt, pc, pre_embs=None, dr=0.0, wdr=0.0):
        super(LSTMDecoder, self).__init__(pc)
        # Store hyperparameters
        self.nl, self.di, self.de, self.dh = nl, di, de, dh
        self.dr, self.wdr = dr, wdr
        self.vt = vt
        # LSTM Encoder
        self.lstm = dy.VanillaLSTMBuilder(self.nl, self.di + self.de, self.dh, self.pc)
        # Linear layer from last encoding to initial state
        self.Wp_p = self.pc.add_parameters((self.di, self.de), name='Wp')
        self.bp_p = self.pc.add_parameters((self.di,), name='bp')
        # Output layer
        self.Wo_p = self.pc.add_parameters((self.di, self.dh + self.de + self.di), name='Wo')
        self.bo_p = self.pc.add_parameters((self.di,), name='bo')
        # Embedding matrix
        if pre_embs is not None:
            self.E_p = self.pc.parameters_from_numpy(pre_embs, name='E')
        else:
            self.E_p = self.pc.add_parameters((self.vt, self.di), name='E')
        self.b_p = self.pc.add_parameters((self.vt,), init=dy.ConstInitializer(0), name='b')

    def init(self, H, y, usr, test=True, update=True):
        bs = len(y[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Initialize first state of the decoder with the last state of the encoder
        self.Wp = self.Wp_p.expr(update)
        self.bp = self.bp_p.expr(update)
        last_enc = dy.pick(H, index=H.dim()[0][-1] - 1, dim=1)
        init_state = dy.affine_transform([self.bp, self.Wp, last_enc])
        init_state = [dy.zeros(self.dh, batch_size=bs), init_state]
        self.ds = self.lstm.initial_state(init_state, update=update)
        # Initialize dropout masks
        if not test:
            self.lstm.set_dropout_masks(bs)

        self.Wo = self.Wo_p.expr(update)
        self.bo = self.bo_p.expr(update)

        self.E = self.E_p.expr(update)
        self.b = self.b_p.expr(update)

    def next(self, w, c, test=True, state=None):
        if isinstance(w, dy.Expression):
            e = w
        else:
            e = dy.pick_batch(self.E, w)

        if not test:
            e = dy.dropout_dim(e, 0, self.wdr)
        x = dy.concatenate([e, c])
        # Run LSTM
        if state is None:
            self.ds = self.ds.add_input(x)
            next_state = self.ds
        else:
            next_state = state.add_input(x)
        h = next_state.output()
        return h, e, next_state

    def s(self, h, c, e, test=True):
        output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([h, c, e])])
        if not test:
            output = dy.dropout(output, self.dr)
        # Score
        s = dy.affine_transform([self.b, self.E, output])
        return s
    
    def load_pretrained(self, filename):
        self.lstm.param_collection().populate(filename, self.lstm.param_collection().name())
        self.Wp_p.populate(filename, self.pc.name() + '/Wp')
        self.bp_p.populate(filename, self.pc.name() + '/bp')
        self.Wo_p.populate(filename, self.pc.name() + '/Wo')
        self.bo_p.populate(filename, self.pc.name() + '/bo')
        self.E_p.populate(filename, self.pc.name() + '/E')
        self.b_p.populate(filename, self.pc.name() + '/b')

class OutLSTMDecoder(Decoder):
    """docstring for LSTMDecoder"""

    def __init__(self, nl, di, de, dh, vt, dt, pc, pre_embs=None, dr=0.0, wdr=0.0):
        super(OutLSTMDecoder, self).__init__(pc)
        # Store hyperparameters
        self.nl, self.di, self.de, self.dh = nl, di, de, dh
        self.dr, self.wdr = dr, wdr
        self.dt = dt
        self.vt = vt
        # LSTM Encoder
        self.lstm = dy.VanillaLSTMBuilder(self.nl, self.di + self.de, self.dh, self.pc)
        # Linear layer from last encoding to initial state
        self.Wp_p = self.pc.add_parameters((self.di, self.de), name='Wp')
        self.bp_p = self.pc.add_parameters((self.di,), name='bp')
        # Output layer
        self.Wo_p = self.pc.add_parameters((self.di, self.dh + self.de + self.di), name='Wo')
        self.To_p = self.pc.add_parameters((self.di, self.dh + self.de + self.di, self.dt), name='To', init=dy.ConstInitializer(0))
        self.bo_p = self.pc.add_parameters((self.di,), name='bo')
        # Embedding matrix
        if pre_embs is not None:
            self.E_p = self.pc.parameters_from_numpy(pre_embs, name='E')
        else:
            self.E_p = self.pc.add_parameters((self.vt, self.di), name='E')
        self.b_p = self.pc.add_parameters((self.vt,), init=dy.ConstInitializer(0), name='b')

    def init(self, H, y, usr, test=True, update=True):
        bs = len(y[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Initialize first state of the decoder with the last state of the encoder
        self.Wp = self.Wp_p.expr(update)
        self.bp = self.bp_p.expr(update)
        last_enc = dy.pick(H, index=H.dim()[0][-1] - 1, dim=1)
        init_state = dy.affine_transform([self.bp, self.Wp, last_enc])
        init_state = [dy.zeros(self.dh, batch_size=bs), init_state]
        self.ds = self.lstm.initial_state(init_state, update=update)
        # Initialize dropout masks
        if not test:
            self.lstm.set_dropout_masks(bs)

        self.Wo = dy.contract3d_1d_bias(self.To_p.expr(update), usr, self.Wo_p.expr(update))
        self.bo = self.bo_p.expr(update)

        self.E = self.E_p.expr(update)
        self.b = self.b_p.expr(False)

    def next(self, w, c, test=True, state=None):
        e = dy.pick_batch(self.E, w)
        if not test:
            e = dy.dropout_dim(e, 0, self.wdr)
        x = dy.concatenate([e, c])
        # Run LSTM
        if state is None:
            self.ds = self.ds.add_input(x)
            next_state = self.ds
        else:
            next_state = state.add_input(x)
        h = next_state.output()
        return h, e, next_state

    def s(self, h, c, e, test=True):
        output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([h, c, e])])
        if not test:
            output = dy.dropout(output, self.dr)
        # Score
        s = dy.affine_transform([self.b, self.E, output])
        return s

    def load_pretrained(self, filename):
        self.lstm.param_collection().populate(filename, self.lstm.param_collection().name())
        self.Wp_p.populate(filename, self.pc.name() + '/Wp')
        self.bp_p.populate(filename, self.pc.name() + '/bp')
        self.Wo_p.populate(filename, self.pc.name() + '/Wo')
        self.bo_p.populate(filename, self.pc.name() + '/bo')
        self.E_p.populate(filename, self.pc.name() + '/E')
        self.b_p.populate(filename, self.pc.name() + '/b')

class InitLSTMDecoder(LSTMDecoder):
    """The InitLSTM decoder uses a special initialization for all users"""

    def __init__(self, nl, di, de, dh, vt, du, pc, pre_embs=None, dr=0.0, wdr=0.0):
        super(InitLSTMDecoder, self).__init__(nl, di, de, dh, vt, pc, pre_embs, dr, wdr)
        # Store hyperparameters
        self.du = du
        # Transform user vector 
        self.Wu_p = self.pc.add_parameters((self.di, self.du), name='Wu')

    def init(self, H, y, usr, test=True, update=True):
        bs = len(y[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Initialize first state of the decoder with the last state of the encoder
        self.Wp = self.Wp_p.expr(update)
        self.bp = self.bp_p.expr(update)
        self.Wu = self.Wu_p.expr(update)
        last_enc = dy.pick(H, index=H.dim()[0][-1] - 1, dim=1)
        init_state = dy.affine_transform([self.bp, self.Wp, last_enc, self.Wu, dy.nobackprop(usr)])
        init_state = [init_state, dy.zeroes((self.dh,), batch_size=bs)]
        self.ds = self.lstm.initial_state(init_state, update=update)
        # Initialize dropout masks
        if not test:
            self.lstm.set_dropout_masks(bs)

        self.Wo = self.Wo_p.expr(update)
        self.bo = self.bo_p.expr(update)

        self.E = self.E_p.expr(update)
        self.b = self.b_p.expr(update)

class VocLSTMDecoder(LSTMDecoder):
    """The VocLSTM decoder uses a special bias for all users"""

    def __init__(self, nl, di, de, dh, vt, du, pc, pre_embs=None, dr=0.0, wdr=0.0):
        super(VocLSTMDecoder, self).__init__(nl, di, de, dh, vt, pc, pre_embs, dr, wdr)
        # Store hyperparameters
        self.du = du
        # User bias
        self.ub_p = self.pc.add_parameters((self.vt, self.du), init=dy.ConstInitializer(0), name='ub')

    def init(self, H, y, usr, test=True, update=True):
        super(VocLSTMDecoder, self).init(H, y, usr, test, update)
        # Init vocab bias
        self.ub = dy.affine_transform([self.b, self.ub_p.expr(update), usr])

    def s(self, h, c, e, test=True):
        output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([h, c, e])])
        if not test:
            output = dy.dropout(output, self.dr)
        # Score
        s = dy.affine_transform([self.ub, self.E, output])
        return s

class FullVocLSTMDecoder(LSTMDecoder):
    """The FullVocLSTM decoder uses a special bias for all users, not factorized (omg)"""

    def init(self, H, y, usr, test=True, update=True):
        super(FullVocLSTMDecoder, self).init(H, y, usr, test, update)
        # Init vocab bias
        self.ub = usr
        self.scores = []

    def s(self, h, c, e, test=True):
        output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([h, c, e])])
        if not test:
            output = dy.dropout(output, self.dr)
        # Score
        self.scores.append(dy.affine_transform([self.b, self.E, output]))
        s = self.scores[-1] + self.ub
        return s

def get_decoder(decoder, nl, di, de, dh, vt, du, pc, pre_embs=None, dr=0.0, wdr=0.0):
    if decoder == 'lm':
        return LSTMLMDecoder(nl, di, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    elif decoder == 'lstm':
        return LSTMDecoder(nl, di, de, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    elif decoder == 'usr_out_lstm':
        return OutLSTMDecoder(nl, di, de, dh, vt, du, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    elif decoder == 'usr_init_lstm':
        return InitLSTMDecoder(nl, di, de, dh, vt, du, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    elif decoder == 'usr_voc_lstm':
        return VocLSTMDecoder(nl, di, de, dh, vt, du, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    elif decoder == 'usr_full_voc_lstm':
        return FullVocLSTMDecoder(nl, di, de, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    else:
        print('Unknown decoder type "%s", using lstm decoder' % decoder)
        return LSTMDecoder(nl, di, de, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
