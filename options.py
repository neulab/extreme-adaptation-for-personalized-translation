from __future__ import print_function, division

import numpy as np
import argparse
import yaml

import sys

parser = argparse.ArgumentParser()
# Dynet parameters
parser.add_argument("--dynet-seed", default=0, type=int)
parser.add_argument("--dynet-mem", default=512, type=int)
parser.add_argument("--dynet-gpu", help="Use dynet with GPU", action="store_true")
parser.add_argument("--dynet-autobatch", default=0, type=int, help="Use dynet autobatching")
# Configuration
parser.add_argument("--config_file", '-c',
                    default=None, type=str)
parser.add_argument("--env", '-e', help="Environment in the config file",
                    default='train', type=str)
# File/folder paths
parser.add_argument("--output_dir", '-od', help="Output directory", type=str, default='.')
parser.add_argument("--temp_dir", '-temp', help="Temp directory", type=str, default='temp')
parser.add_argument("--train_usr", '-tt', help="List of talks for the training data", type=str)
parser.add_argument("--valid_usr", '-dt', help="List of talks for the validation data", type=str)
parser.add_argument("--test_usr", '-tet', help="List of talks for the test data", type=str)
parser.add_argument("--train_src", '-ts', help="Train data in the source language", type=str)
parser.add_argument("--train_trg", '-td', help="Train data in the target language", type=str)
parser.add_argument("--valid_src", '-vs', help="Validation data in the source language", type=str)
parser.add_argument("--valid_trg", '-vd', help="Validation data in the target language", type=str)
parser.add_argument("--test_src", '-tes', help="Test data in the source language", type=str)
parser.add_argument("--test_trg", '-ted', help="Test data in the target language", type=str)
parser.add_argument("--lex_file", '-lf',
                    help="File to save the target language dictionary to", type=str)
parser.add_argument("--lex_s2t", '-ls2t',
                    help="File containing a lexicon from source to target", type=str)
parser.add_argument("--lex_t2s", '-lt2s',
                    help="File containing a lexicon from target to source", type=str)
parser.add_argument("--pretrained_wembs", '-prew',
                    help="File containing pretrained word embeddings", type=str)
parser.add_argument("--pretrained_user", '-preusr',
                    help="File containing pretrained user embeddings", type=str)
parser.add_argument("--test_out", '-teo', help="File to save the translated test data", type=str)
parser.add_argument("--valid_out", '-vo',
                    help="File to save the translated validation data", type=str)
parser.add_argument("--lm_file", '-lmf', help="File to save the target language model", type=str)
parser.add_argument("--model", '-m', type=str,
                    help='Model file ([exp_name]_model if not specified)')
parser.add_argument("--usr_file_list", '-ufl', type=str,
                    help='File containing a list of user data file names')

# Hyper-parameters
parser.add_argument('--n_test', type=int, default=5,
                    help='How many sentences to use of evaluating for each user')
parser.add_argument('--max_n_train', type=int, default=5,
                    help='Maximum number of sentences to train on for new users')
parser.add_argument('--min_n_train', type=int, default=1,
                    help='Minimum number of sentences to train on for new users')
parser.add_argument("--trainer", '-tr', type=str,
                    help='Optimizer. Choose from "sgd,clr,momentum,adam,rmsprop"', default='sgd')
parser.add_argument('--num_epochs', '-ne', type=int, default=1,
                    help='Number of epochs (full pass over the training data) to train on')
parser.add_argument('--patience', '-p', type=int, default=0,
                    help='Patience before early stopping. No early stopping if <= 0')
parser.add_argument('--usr_onehot', '-toh', type=int, default=-1,
                    help='Use talk 1hot vector')
parser.add_argument('--src_vocab_size', '-svs',
                    type=int, help='Maximum vocab size of the source language', default=40000)
parser.add_argument('--trg_vocab_size', '-tvs',
                    type=int, help='Maximum vocab size of the target language', default=20000)
parser.add_argument('--batch_size', '-bs',
                    type=int, help='minibatch size', default=20)
parser.add_argument('--dev_batch_size', '-dbs',
                    type=int, help='minibatch size for the validation set', default=10)
parser.add_argument("--encoder", '-enc', type=str,
                    help='Encoder type', default='bilstm')
parser.add_argument("--attention", '-att', type=str,
                    help='Attention type', default='mlp')
parser.add_argument("--decoder", '-dec', type=str,
                    help='Encoder type', default='lstm')
parser.add_argument("--user_recognizer", '-usr', type=str,
                    help='user recognizer type', default='lookup')
parser.add_argument("--user_token", help='user token', action="store_true")
parser.add_argument("--update_mode", '-um', type=str,
                    choices=['full', 'biases', 'mixture_weights'],
                    help='Update mode for new users', default='full')
parser.add_argument('--num_layers', '-nl', type=int, default=1,
                    help='Number of layers in the encoder/decoder (For now only one is supported)')
parser.add_argument('--emb_dim', '-de',
                    type=int, help='Embedding dimension', default=256)
parser.add_argument('--att_dim', '-da',
                    type=int, help='Attention dimension', default=256)
parser.add_argument('--hidden_dim', '-dh',
                    type=int, help='Hidden dimension (for the recurrent networks)', default=256)
parser.add_argument('--usr_dim', '-dusr',
                    type=int, help='User dimension', default=10)
parser.add_argument('--label_smoothing', '-ls', type=float, default=0.0,
                    help='Label smoothing (interpolation coefficient with '
                    'the uniform distribution)')
parser.add_argument('--language_model', '-lm',
                    type=str, help='Language model to interpolate with', default=None)
parser.add_argument('--dropout_rate', '-dr',
                    type=float, help='Dropout rate', default=0.0)
parser.add_argument('--word_dropout_rate', '-wdr',
                    type=float, help='Word dropout rate', default=0.0)
parser.add_argument('--gradient_clip', '-gc', type=float, default=1.0,
                    help='Gradient clipping. Negative value means no clipping')
parser.add_argument('--learning_rate', '-lr',
                    type=float, help='learning rate', default=1.0)
parser.add_argument('--learning_rate_decay', '-lrd',
                    type=float, help='learning rate decay', default=0.0)
parser.add_argument('--check_train_error_every', '-ct',
                    type=int, help='Check train error every', default=100)
parser.add_argument('--check_valid_error_every', '-cv',
                    type=int, help='Check valid error every', default=1000)
parser.add_argument('--valid_bleu_every', '-vbe',
                    type=int, help='Compute BLEU on validation set every', default=500)
parser.add_argument('--max_len', '-ml', type=int,
                    help='Maximum length of generated sentences', default=60)
parser.add_argument('--beam_size', '-bm', type=int,
                    help='Beam size for beam search', default=1)
parser.add_argument('--bootstrap_num_samples', '-M', type=int, default=100,
                    help='Number of samples for bootstrap resampling')
parser.add_argument('--bootstrap_sample_size', type=float, default=50,
                    help='Size of each sample (in percentage of the total size)')
parser.add_argument('--min_freq', '-mf', type=int,
                    help='Minimum frequency under which words are unked', default=1)
parser.add_argument("--exp_name", '-en', type=str, default='experiment',
                    help='Name of the experiment (used so save the model)')
parser.add_argument("--verbose", '-v',
                    help="increase output verbosity",
                    action="store_true")
parser.add_argument("--train",
                    help="Print debugging info",
                    action="store_true")
parser.add_argument("--test",
                    help="Print debugging info",
                    action="store_true")
parser.add_argument("--pretrained",
                    help="Whether to use a pretrained model",
                    action="store_true")
parser.add_argument("--retranslate",
                    help="Whether to retranslate the test data (true by default)",
                    action="store_false")
parser.add_argument("--full_training",
                    help="Update all parameters when performing user adaptation",
                    action="store_true")
parser.add_argument("--user_training",
                    help="Only train user recognizer/user specific part",
                    action="store_true")
parser.add_argument("--unk_replacement",
                    help="UNK replacement",
                    action="store_true")
parser.add_argument("--log_unigram_bias",
                    help="Add a log-unigram bias term",
                    action="store_true")
parser.add_argument("--use_trg_unigrams",
                    help="use target unigrams directly",
                    action="store_true")


def parse_options():
    """Parse options from command line arguments and optionally config file

    Returns:
        Options
        argparse.Namespace
    """
    opt = parser.parse_args()
    if opt.config_file:
        with open(opt.config_file, 'r') as f:
            data = yaml.load(f)
            delattr(opt, 'config_file')
            arg_dict = opt.__dict__
            for key, value in data.items():
                if isinstance(value, dict):
                    if key == opt.env:
                        for k, v in value.items():
                            arg_dict[k] = v
                    else:
                        continue
                else:
                    arg_dict[key] = value
    # Little trick : add dynet general options to sys.argv if they're not here
    # already. Linked to this issue : https://github.com/clab/dynet/issues/475
    #sys.argv.append('--dynet-devices')
    #sys.argv.append('CPU,GPU:0')
    if opt.dynet_gpu and '--dynet-gpus' not in sys.argv:
        sys.argv.append('--dynet-gpus')
        sys.argv.append('1')
    if '--dynet-autobatch' not in sys.argv:
        sys.argv.append('--dynet-autobatch')
        sys.argv.append(str(opt.__dict__['dynet_autobatch']))
    if '--dynet-mem' not in sys.argv:
        sys.argv.append('--dynet-mem')
        sys.argv.append(str(opt.__dict__['dynet_mem']))
    if '--dynet-seed' not in sys.argv:
        sys.argv.append('--dynet-seed')
        sys.argv.append(str(opt.__dict__['dynet_seed']))
        if opt.__dict__['dynet_seed'] > 0:
            np.random.seed(opt.__dict__['dynet_seed'])
    return opt


def print_config(opt, **kwargs):
    """Print the current configuration

    Prints command line arguments plus any kwargs

    Arguments:
        opt (argparse.Namespace): Command line arguments
        **kwargs: Any other key=value pair
    """
    print('======= CONFIG =======')
    for k, v in vars(opt).items():
        print(k, ':', v)
    for k, v in kwargs.items():
        print(k, ':', v)
    print('======================')


# Do this so sys.argv is changed upon import
options = parse_options()


def get_options():
    """Clean way to get options

    Returns:
        Options
        argparse.Namespace
    """
    return options
