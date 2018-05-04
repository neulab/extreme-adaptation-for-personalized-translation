from __future__ import print_function, division

import numpy as np
import dynet as dy

import encoders
import attention
import decoders
import user
import beam

import sys


class Seq2SeqModel(object):
    """A neural sequence to sequence model with attention

    Uses LSTM encoder and decoder, as well as tanh based attention

    Extends:
        object
    """

    def __init__(self,
                 opt,
                 lexicon,
                 pretrained_wembs=None,
                 pretrained_user=None,
                 lang_model=None):
        """Constructor

        :param input_dim: Embedding dimension
        :param hidden_dim: Dimension of the recurrent layers
        :param att_dim: Dimension of the hidden layer in the attention MLP
        :param lexicon: Lexicon object containing dictionaries/bilingual discrete lexicons...
        :param enc_type: Type of encoder
        :param att_type: Type of attention mechanism
        :param dec_type: Type of decoder
        :param model_file: File where the model should be saved (default: (None))
        :param label_smoothing: interpolation coefficient with second output distribution
        :param dropout: dropout rate for parameters
        :param word_dropout: dropout rate for words in the decoder
        :param max_len: Maximum length allowed when generating translations (default: (60))
        """
        # Store config
        self.nl = opt.num_layers
        self.dr, self.wdr = opt.dropout_rate, opt.word_dropout_rate
        self.ls, self.ls_eps = (opt.label_smoothing > 0), opt.label_smoothing
        self.max_len = opt.max_len
        self.src_sos, self.src_eos = lexicon.w2ids['SOS'], lexicon.w2ids['EOS']
        self.trg_sos, self.trg_eos = lexicon.w2idt['SOS'], lexicon.w2idt['EOS']
        # Dimensions
        self.vs, self.vt = len(lexicon.w2ids), len(lexicon.w2idt)
        self.du = opt.usr_dim
        self.nu = len(lexicon.usr2id)
        self.di, self.dh, self.da = opt.emb_dim, opt.hidden_dim, opt.att_dim
        # Model
        self.pc = dy.ParameterCollection()
        self.model_file = opt.model
        # Encoder
        self.enc = encoders.get_encoder(opt.encoder, self.nl, self.di,
                                        self.dh, self.du, self.vs, self.pc,
                                        dr=self.dr, pre_embs=pretrained_wembs)
        # Attention module
        self.att = attention.get_attention(opt.attention, self.enc.dim, self.dh, self.da, self.pc)
        # Decoder
        self.dec = decoders.get_decoder(opt.decoder, self.nl, self.di,
                                        self.enc.dim, self.dh, self.vt, self.du,
                                        self.pc, pre_embs=pretrained_wembs, dr=self.dr, wdr=self.wdr)
        # User recognizer parameters
        self.usr = user.ZeroVocabUserRecognizer(self.vt, self.nu, self.enc.dim, self.pc)
        # Target language model (for label smoothing)
        self.lm = lang_model
        self.lex = lexicon
        self.unk_replace = opt.unk_replacement
        self.user_token = opt.user_token
        self.test = True
        self.update = True
    
    def set_usr(self, usr_type, pretrained_user=None):
        self.usr = user.get_user_recognizer(usr_type, self.nu, self.du, self.enc.dim, self.vt, self.pc, pre_user=pretrained_user)

    def set_test_mode(self):
        self.test = True

    def set_train_mode(self):
        self.test = False

    def freeze_parameters(self):
        self.update = False

    def thaw_parameters(self):
        self.update = True

    def prepare_batch(self, batch, eos):
        """Prepare batch of sentences for sequential processing

        Basically transposes the batch, pads sentences of different lengths
            with EOS symbols and builds a mask for the loss function
            (so that the loss is masked on the padding words).

        Example (with strings instead of int for clarity):

        [["I","like","chocolate"],["Me","too"]]
        -> [["I","Me"],["like","too"],["chocolate","EOS"]], [[1,1],[1,1],[1,0]]

        :param batch: List of sentences
        :param eos: EOS index

        :returns: (prepared_batch, masks) both of shape (sentence_length, batch_size)
        """
        bsize = len(batch)

        batch_len = max(len(s) for s in batch)

        x = np.zeros((batch_len, bsize), dtype=int)
        masks = np.ones((batch_len, bsize), dtype=float)
        x[:] = eos

        for i in range(bsize):
            sent = batch[i][:]
            masks[len(sent):, i] = 0.0
            while len(sent) < batch_len:
                sent.append(eos)
            x[:, i] = sent
        return x, masks

    def set_usr_vec(self, usr, test=True):
        self.usr_vec = dy.softmax(dy.lookup_batch(self.Ws_p, usr))
        # if not test:
        #    self.usr_vec = dy.dropout(self.usr_vec, self.dr)

    def reset_usr_vec(self):
        self.usr.reset(0)

    def encode(self, src, test=False):
        """Encode a batch of sentences

        :param src: List of sentences. It is assumed that all
            source sentences have the same length

        :returns: Expression of the encodings
        """
        # Prepare batch
        x, _ = self.prepare_batch(src, self.src_eos)
        self.enc.init(x, self.usr.user_vector, test=self.test, update=self.update)
        return self.enc(x, test=self.test, update=self.update)

    def attend(self, encodings, h):
        """Compute attention score

        Given :math:`z_i` the encoder's output at time :math:`i`, :math:`h_{j-1}`
        the decoder's output at time :math:`j-1`, the attention score is computed as :

        .. math::

            \begin{split}
                s_{ij}&=V_a^T\tanh(W_az_i + W_{ha}h_j + b_a)\\
                \alpha_{ij}&=\frac{s_{ij}}{\sum_{i'}s_{i'j}}\\
            \end{split}

        :param encodings: Source sentence encodings obtained with self.encode
        :param h: Decoder output at the previous timestep

        :returns: Two dynet Expressions, the context and the attention weights
        """
        self.att.init(test=self.test, update=self.update)
        return self.att(encodings, h, test=self.test)

    def cross_entropy_loss(self, s, nw, cw):
        """Calculates the cross-entropy
        """
        if self.ls:
            log_prob = dy.log_softmax(s)
            if self.lm is None:
                loss = - dy.pick_batch(log_prob, nw) * (1 - self.ls_eps) - \
                    dy.mean_elems(log_prob) * self.ls_eps
            else:
                loss = - dy.pick_batch(log_prob, nw) * (1 - self.ls_eps) - \
                    dy.dot_product(self.lm_e, log_prob) * self.ls_eps
        else:
            loss = dy.pickneglogsoftmax_batch(s, nw)
        return loss

    def decode_loss(self, encodings, trg):
        """Compute the negative conditional log likelihood of the target sentence
        given the encoding of the source sentence

        :param encodings: Source sentence encodings obtained with self.encode
        :param trg: List of target sentences

        :returns: Expression of the loss averaged on the minibatch
        """
        y, masksy = self.prepare_batch(trg, self.trg_eos)
        slen, bsize = y.shape
        # Init decoder
        self.dec.init(encodings, y, self.usr.user_vector, test=self.test, update=self.update)
        # Initialize context
        context = dy.zeroes((self.enc.dim,), batch_size=bsize)
        # Process user token if necessary
        if self.user_token:
            _, _, _ = self.dec.next(self.usr.user_vector, context, test=self.test)
        # Start decoding
        errs = []
        for cw, nw, mask in zip(y, y[1:], masksy[1:]):
            # Run LSTM
            h, e, _ = self.dec.next(cw, context, test=self.test)
            # Compute next context
            context, _ = self.attend(encodings, h)
            # Score
            s = self.dec.s(h, context, e, test=self.test)
            masksy_e = dy.inputTensor(mask, batched=True)
            # Loss
            loss = self.cross_entropy_loss(s, nw, cw)
            loss = dy.cmult(loss, masksy_e)
            errs.append(loss)
        # Add all losses together
        err = dy.mean_batches(dy.esum(errs))
        return err

    def user_loss(self, encodings, usr):
        return self.usr.predict_loss(encodings, usr)

    def calculate_loss(self, src, trg, usr, test=False):
        """Compute the conditional log likelihood of the target sentences given the source sentences

        Combines encoding and decoding

        :param src: List of sentences. It is assumed that all
                    source sentences have the same length
        :param trg: List of target sentences

        :returns: Expression of the loss averaged on the minibatch
        """
        dy.renew_cg()
        self.usr.init(src, usr, test=self.test, update=self.update)
        if self.lm is not None:
            self.lm.init()
        encodings = self.encode(src)
        err = self.decode_loss(encodings, trg)
        return err

    def calculate_user_loss(self, src, trg, usr=[0], test=False, update_mode='full'):
        if update_mode == 'full':
            self.thaw_parameters()
        else:
            self.freeze_parameters()
        self.usr.init(src, usr, test=self.test, update=self.update, update_mode=update_mode)
        if self.lm is not None:
            self.lm_e = dy.inputTensor(self.lm)
        encodings = self.encode(src)
        #user_nll = self.user_loss(encodings, usr)
        err = self.decode_loss(encodings, trg)
        return err

    def precompute_scores(self, src, trg, usr=[0], test=False):
        self.usr.init(src, usr, test=self.test, update=self.update, update_mode=update_mode)
        encodings = self.encode(src)
        #user_nll = self.user_loss(encodings, usr)
        err = self.decode_loss(encodings, trg)
        return self.decoder.scores
    
    def calculate_user_bias_loss(self, scores, trg, usr=[0], test=False, update_mode='full'):
        self.usr.init(None, usr, test=self.test, update=self.update, update_mode=update_mode)
        if self.lm is not None:
            self.lm_e = dy.inputTensor(self.lm)
        #user_nll = self.user_loss(encodings, usr)
        errs = []
        user_bias = self.usr.user_vector
        for score, w in zip(scores[0], trg[0][1:]):
            s = dy.inputTensor(score) + user_bias 
            err = self.cross_entropy_loss(s, [w], None)
            errs.append(err)

        return dy.esum(errs)

    def precompute_scores(self, src, trg, usr=[0], test=False):
        self.usr.init(src, usr, test=self.test, update=self.update, update_mode='full')
        if self.lm is not None:
            self.lm_e = dy.inputTensor(self.lm)
        encodings = self.encode(src)
        #user_nll = self.user_loss(encodings, usr)
        err = self.decode_loss(encodings, trg)
        return self.dec.scores
    
    def translate(self, src, usr, beam_size=1, usr_onehot=-1):
        """Translate a source sentence

        Translate a single source sentence by decoding using beam search

        :param src: Source sentence (list of strings)
        :param beam_size: Size of the beam for beam search.
            A value of 1 means greedy decoding (default: (1))

        :returns generated translation (list of indices)
        """
        dy.renew_cg()
        x = self.lex.sent_to_ids(src)
        if usr_onehot < 0:
            self.usr.init([x], [usr])
        else:
            onehot = np.zeros(self.du)
            onehot[usr_onehot] = 1
            self.usr.usr_vec = dy.inputTensor(onehot)
        input_len = len(x)
        encodings = self.encode([x], test=True)
        # Decode
        b = self.beam_decode(encodings, input_len=len(x), beam_size=beam_size)
        # Post process (unk replacement...)
        sent = self.post_process(b, src)
        return ' '.join(sent)

    def beam_decode(self, encodings, input_len=10, beam_size=1):
        # Add parameters to the graph
        self.dec.init(encodings, [[self.trg_sos]], self.usr.user_vector,
                      test=self.test, update=self.update)
        # Initialize context
        context = dy.zeroes((self.enc.dim,))
        # Process user token if necessary
        if self.user_token:
            _, _, _ = self.dec.next(self.usr.user_vector, context, test=self.test)
        # Get conditional log probability of lengths
        llp = np.log(self.lex.p_L[input_len])
        # Initialize beam
        beams = [beam.Beam(self.dec.ds, context, [self.trg_sos], llp[0])]
        # Loop
        for i in range(int(min(self.max_len, input_len * 1.5))):
            new_beam = []
            for b in beams:
                if b.words[-1] == self.trg_eos:
                    new_beam.append(beam.Beam(b.state, b.context, b.words, b.logprob, b.align))
                    continue
                h, e, b.state = self.dec.next([b.words[-1]], b.context, state=b.state)
                # Compute next context
                b.context, att = self.attend(encodings, h)
                # Score
                s = self.dec.s(h, b.context, e, test=self.test)
                # Probabilities
                p = dy.softmax(s).npvalue()
                # Careful for floating errors
                p = p.flatten() / p.sum()
                # Store alignment for e.g. unk replacement
                align = np.argmax(att.npvalue())
                kbest = np.argsort(p)
                for nw in kbest[-beam_size:]:
                    new_beam.append(beam.Beam(b.state, b.context,
                                              b.words + [nw],
                                              b.logprob + np.log(p[nw]) + llp[i + 1] - llp[i],
                                              b.align + [align]))
            # Only keep the best
            beams = sorted(new_beam, key=lambda b: b.logprob)[-beam_size:]
            if beams[-1].words[-1] == self.trg_eos:
                break

        return beams[-1]

    def post_process(self, b, src):
        sent = self.lex.ids_to_sent(b.words, trg=True)
        if self.unk_replace:
            for i, w in enumerate(sent):
                if w == 'UNK':
                    sent[i] = self.lex.translate(src[b.align[i + 1]])
        return sent

    def save(self):
        """Save model

        Saves the model holding the parameters to self.model_file
        """
        self.pc.save(self.model_file)

    def load(self):
        """Load model

        Loads the model holding the parameters from self.model_file
        """
        self.pc.populate(self.model_file)

    def load_user_agnostic(self):
        self.enc.pc.populate(self.model_file, '/enc')
        self.att.pc.populate(self.model_file, '/att')
        self.dec.load_pretrained(self.model_file)
