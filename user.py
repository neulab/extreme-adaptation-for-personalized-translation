from __future__ import print_function, division

import numpy as np
import dynet as dy

import sys

class UserRecognizer(object):

    def __init__(self, pc):
        self.pc = pc.add_subcollection('usr')

    def init(self, x, usr, test=True, update=True):
        pass

    @property
    def user_vector(self):
        raise NotImplemented()
    
    def reset(self, index):
        pass
    
    def __call__(self, x, usr, test=True):
        raise NotImplemented()

class EmptyUserRecognizer(UserRecognizer):
    """Returns None all the time
    
    This class is used for the baseline model where no user information is needed/provided
    """
    def __call__(self, x, usr, test=True):
        return None
    
    @property
    def user_vector(self):
        return None

class LookupUserRecognizer(UserRecognizer):
    """Stores a lookup table with user vectors
    """

    def __init__(self, du, nu, de, pc, pre_user=None):
        super(LookupUserRecognizer, self).__init__(pc)


        self.du, self.nu = du, nu
        if pre_user is None:
            init = dy.NormalInitializer(1 / self.du, np.sqrt(1 / self.du))
            self.U_p = self.pc.add_lookup_parameters((self.nu, self.du), init=init, name='U')
        else:
            self.U_p = self.pc.lookup_parameters_from_numpy(pre_user, name='U')

    def init(self, x, usr, test=True, update=True, update_mode='full'):
        if update_mode=='biases':
            self.usr_vec = dy.lookup_batch(self.U_p, usr, True)
        else:
            self.usr_vec = dy.lookup_batch(self.U_p, usr, update)
    
    def __call__(self, x, usr):
        return self.usr_vec
    
    def predict_loss(self, encodings, usr):
        avg_enc = dy.mean_dim(encodings, 1)
        h = dy.rectify(dy.affine_transform([self.bh, self.Wh, avg_enc]))
        s = dy.affine_transform([self.bu, self.Wu, h])
        return dy.mean_batches(dy.squared_distance(s, self.usr_vec))
    
    def reset(self, index, value=None):
        if value is None:
            value = np.ones(self.du) / self.du
        self.U_p.init_row(index, value)

    @property
    def user_vector(self):
        return self.usr_vec

class SoftmaxLookupUserRecognizer(LookupUserRecognizer):
    
    def init(self, x, usr, test=True, update=True):
        self.usr_vec = dy.softmax(dy.lookup_batch(self.U_p, usr, update=update))

class ZeroVocabUserRecognizer(UserRecognizer):
    """Stores a vocabulary bias vector for each user
    """

    def __init__(self, v, nu, de, pc, pretrained_BU=None):
        super(ZeroVocabUserRecognizer, self).__init__(pc)

        # prediction parameters
        self.nu, self.v = nu, v
        if pretrained_BU is None:
            init = dy.ConstInitializer(0)
            self.B_p = self.pc.add_parameters(self.v,init=init, name='B')
        else:
            self.B_p = self.pc.parameter_from_numpy(pretrained_BU, name='B')

        self.val = None

    def init(self, x, usr, test=True, update=True, update_mode='full'):
        if update_mode == 'biases':
            self.usr_vec = self.B_p.expr(True)
        else:
            self.usr_vec = self.B_p
    
    def __call__(self, x, usr):
        return self.usr_vec
    
    def predict_loss(self, encodings, usr):
        return dy.zeros(1)

    @property
    def user_vector(self):
        return self.usr_vec
    
    def reset(self, index, value=None):
        #if value is None:
        #    if self.avg is None:
        #        self.avg = self.BU_p.as_array().mean(axis=0)
        #    value = seplf.avg
        if self.val is None:
            self.val = self.B_p.as_array()
        self.B_p.set_value(self.val)#init_row(index, value)

class FullVocabUserRecognizer(UserRecognizer):
    """Stores a vocabulary bias vector for each user
    """

    def __init__(self, v, nu, de, pc, pretrained_BU=None):
        super(FullVocabUserRecognizer, self).__init__(pc)

        # prediction parameters
        self.Wh_p = self.pc.add_parameters((de, de), name='Wh')
        self.bh_p = self.pc.add_parameters((de,), name='bh', init=dy.ConstInitializer(0))
        self.Su_p = self.pc.add_parameters((nu, de), name='Su')
        self.bu_p = self.pc.add_parameters((nu,), name='bu', init=dy.ConstInitializer(0))

        self.v, self.nu = v, nu
        if pretrained_BU is None:
            init = dy.ConstInitializer(0)
            self.BU_p = self.pc.add_lookup_parameters((self.nu, self.v),init=init, name='BU')
        else:
            self.BU_p = self.pc.lookup_parameter_from_numpy(pretrained_BU, name='BU')
        self.avg = None

    def init(self, x, usr, test=True, update=True, update_mode='full'):
        self.Wh = self.bh_p
        self.bh = self.bh_p
        self.Su = self.Su_p
        self.bu = self.bu_p
        if update_mode=='biases':
            self.usr_vec = dy.lookup_batch(self.BU_p, usr, True)
        else:
            self.usr_vec = dy.lookup_batch(self.BU_p, usr, update)
    
    def __call__(self, x, usr):
        return self.usr_vec
    
    def predict_loss(self, encodings, usr):
        avg_enc = dy.mean_dim(encodings, 1)
        h = dy.rectify(dy.affine_transform([self.bh, self.Wh, avg_enc]))
        s = dy.affine_transform([self.bu, self.Wu, h])
        return dy.mean_batches(dy.pickneglogsoftmax(s, usr))

    @property
    def user_vector(self):
        return self.usr_vec
    
    def reset(self, index, value=None):
        if value is None:
            if self.avg is None:
                self.avg = self.BU_p.as_array().mean(axis=0)
            value = self.avg
        #self.BU_p.zero()#init_row(index, value)
        self.BU_p.init_row(0, value)#np.zeros(self.v))#init_row(index, value)

class FactVocabUserRecognizer(UserRecognizer):
    """Stores a vocabulary bias vector for each user
    """

    def __init__(self, v, du, nu, de, pc, pretrained_BU=None):
        super(FactVocabUserRecognizer, self).__init__(pc)

        # prediction parameters
        self.Wh_p = self.pc.add_parameters((de, de), name='Wh')
        self.bh_p = self.pc.add_parameters((de,), name='bh', init=dy.ConstInitializer(0))
        self.Su_p = self.pc.add_parameters((du, de), name='Su')
        self.bu_p = self.pc.add_parameters((du,), name='bu', init=dy.ConstInitializer(0))
        self.du = du
        self.v, self.nu = v, nu
        # User vectors
        self.U_p = self.pc.add_lookup_parameters((nu, du), init=dy.ConstInitializer(0), name='U')
        init = dy.NormalInitializer(1 / self.du, np.sqrt(1 / self.du))
        # Biases
        self.B_p = self.pc.add_parameters((v, du), init=init, name='B')
        self.avg = None
        self.BU_p = None

    def init(self, x, usr, test=True, update=True, update_mode='full'):
        self.Wh = self.bh_p
        self.bh = self.bh_p
        self.Su = self.Su_p
        self.bu = self.bu_p
        if update_mode=='biases':
            #self.usr_vec = self.BU_p.expr(True)#dy.pick(self.B_p.expr(True), index=0, dim=1)# * dy.lookup_batch(self.U_p, usr, True)
            self.usr_vec = self.B_p.expr(True) * dy.lookup_batch(self.U_p, usr, True)
        elif update_mode=='mixture_weights':
            self.usr_vec = self.B_p * dy.lookup_batch(self.U_p, usr, True)
        else:
            self.usr_vec = self.B_p * dy.lookup_batch(self.U_p, usr, update)
    
    def __call__(self, x, usr):
        return self.usr_vec
    
    def predict_loss(self, encodings, usr):
        avg_enc = dy.mean_dim(encodings, 1)
        h = dy.rectify(dy.affine_transform([self.bh, self.Wh, avg_enc]))
        s = dy.affine_transform([self.bu, self.Wu, h])
        return dy.mean_batches(dy.pickneglogsoftmax(s, usr))

    @property
    def user_vector(self):
        return self.usr_vec
    
    def reset(self, index, value=None):
        if value is None:
            if self.avg is None:
                self.avg = self.U_p.as_array().mean(axis=0)
            value = self.avg
        self.U_p.init_row(0, value)#np.ones(self.du)/np.sqrt(self.du))#init_row(index, value)
        #if self.BU_p is None:
        #    self.BU_p = self.pc.add_parameters(self.v)
        #self.BU_p.zero()

class LogFactVocabUserRecognizer(UserRecognizer):
    """Stores a vocabulary bias vector for each user
    """

    def __init__(self, v, du, nu, de, pc, pretrained_BU=None):
        super(LogFactVocabUserRecognizer, self).__init__(pc)

        self.du = du
        self.v, self.nu = v, nu
        # User vectors
        self.U_p = self.pc.add_lookup_parameters((nu, 1, du), init=dy.ConstInitializer(0), name='U')
        init = dy.NormalInitializer(1 / self.du, np.sqrt(1 / self.du))
        # Biases
        self.B_p = self.pc.add_parameters((v, du), init=init, name='B')
        self.avg = None
        self.BU_p = None

    def init(self, x, usr, test=True, update=True, update_mode='full'):
        if update_mode=='biases':
            self.usr_vec = dy.logsumexp_dim(self.B_p.expr(True) + dy.lookup_batch(self.U_p, usr, True), d=1)
        elif update_mode=='mixture_weights':
            self.usr_vec = dy.logsumexp_dim(self.B_p.expr(update) + dy.lookup_batch(self.U_p, usr, True), d=1)
        else:
            self.usr_vec = dy.logsumexp_dim(self.B_p.expr(update) + dy.lookup_batch(self.U_p, usr, update), d=1)
    
    def __call__(self, x, usr):
        return self.usr_vec
    
    def predict_loss(self, encodings, usr):
        avg_enc = dy.mean_dim(encodings, 1)
        h = dy.rectify(dy.affine_transform([self.bh, self.Wh, avg_enc]))
        s = dy.affine_transform([self.bu, self.Wu, h])
        return dy.mean_batches(dy.pickneglogsoftmax(s, usr))
    
    @property
    def user_vector(self):
        return self.usr_vec
    
    def reset(self, index, value=None):
        if value is None:
            if self.avg is None:
                self.avg = self.U_p.as_array().mean(axis=0)
            value = self.avg
        self.U_p.init_row(0, value)#np.ones(self.du)/np.sqrt(self.du))#init_row(index, value)
        #if self.BU_p is None:
        #    self.BU_p = self.pc.add_parameters(self.v)
        #self.BU_p.zero()

def get_user_recognizer(usr_rec_type, nu, du, de, v, pc, pre_user=None):

    if usr_rec_type == 'empty':
        return EmptyUserRecognizer(pc)
    elif usr_rec_type == 'lookup':
        return LookupUserRecognizer(du, nu, de, pc, pre_user=pre_user)
    elif usr_rec_type == 'zero_voc':
        return ZeroVocabUserRecognizer(v, nu, de, pc)
    elif usr_rec_type == 'full_voc':
        return FullVocabUserRecognizer(v, nu, de, pc)
    elif usr_rec_type == 'fact_voc':
        return FactVocabUserRecognizer(v, du, nu, de, pc)
    elif usr_rec_type == 'log_fact_voc':
        return LogFactVocabUserRecognizer(v, du, nu, de, pc)
    else:
        print('Unknown user recognizer %s, using EmptyUserRecognizer instead' % usr_rec_type, file=sys.stderr)
        return EmptyUserRecognizer(pc)
