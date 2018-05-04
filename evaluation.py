from __future__ import print_function, division

import subprocess
import argparse
import numpy as np
import os

import sys

from utils import loadtxt, savetxt


def bleu_score(ref_file, hyp_file):
    """Computes corpus level BLEU score with Moses' multi-bleu.pl script

    Arguments:
        ref_file (str): Path to the reference file
        hyp_file (str): Path to the hypothesis file

    Returns:
        tuple: Tuple (BLEU, details) containing the bleu score
            and the detailed output of the perl script

    Raises:
        ValueError: Raises error if the perl script fails for some reason
    """
    command = 'perl scripts/multi-bleu.pl ' + ref_file + ' < ' + hyp_file
    c = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    details, error = c.communicate()
    details, error = details.decode('utf-8'), error.decode('utf-8')
    if not details.startswith('BLEU ='):
        raise ValueError('Error in BLEU score computation:\n%s' % error)
    else:
        BLEU_str = details.split(' ')[2][:-1]
        BLEU = float(BLEU_str)
        return BLEU, details


parser = argparse.ArgumentParser(
    description='program to compare mt results',
)
parser.add_argument('ref', type=str, help='a path to a correct reference file')
parser.add_argument('out', type=str, help='a path to a system output')
parser.add_argument('otherout', nargs='?', type=str, default=None,
                    help='a path to another system output. add only if '
                    'you want to compare outputs from two systems.')
parser.add_argument('--num_samples', '-M', type=int, default=100,
                    help='Number of samples for bootstrap resampling')
parser.add_argument('--sample_size', type=float, default=50,
                    help='Size of each sample (in percentage of the total size)')
parser.add_argument('--bleufile', type=str, default='bleus.txt',
                    help='Where to store the bleu scores')
parser.add_argument("--verbose", '-v',
                    help="increase output verbosity",
                    action="store_true")


def print_stats(bleus):
    print('Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f' %
          (bleus.mean(), bleus.std(), bleus.min(), bleus.max()))
    print('95%% confidence interval: %.3f - %.3f' %
          (bleus[int(0.025 * len(bleus))], bleus[int(0.975 * len(bleus))]))

def print_paired_stats(bleus):
    win1 = (bleus[:, 0] > bleus[:, 1]).sum() / args.num_samples * 100
    win2 = (bleus[:, 0] < bleus[:, 1]).sum() / args.num_samples * 100
    ties = (bleus[:, 0] == bleus[:, 1]).sum() / args.num_samples * 100
    print('System 1 > system 2: %.3f' % win1)
    print('System 1 < system 2: %.3f' % win2)
    print('Ties: %.3f' % ties)

def load_if_file(corpus):
    if isinstance(corpus, str):
        corpus = loadtxt(corpus)
    return corpus

def bootstrap_resampling(ref, out, num_samples, sample_percent, dummy_ref, dummy_out, verbose=False):
    ref = load_if_file(ref)
    out = load_if_file(out)
    n = len(ref)
    assert n == len(out), 'Mismatched reference and output file size'
    k = int(sample_percent * n / 100)
    bleus = []
    for i in range(num_samples):
        subset = np.random.choice(n, k)
        savetxt(dummy_out, out[subset])
        savetxt(dummy_ref, ref[subset])
        bleu, _ = bleu_score(dummy_ref, dummy_out)
        bleus.append(bleu)
        if verbose and (i + 1) % (num_samples // 10) == 0:
            print('%d%% done' % ((i + 1) // (num_samples // 10) * 10))
            sys.stdout.flush()
    bleus = np.sort(np.asarray(bleus))
    return bleus

def paired_bootstrap_resampling(ref, out, otherout, num_samples, sample_percent, dummy_ref, dummy_out, dummy_otherout, verbose=False):
    ref = load_if_file(ref)
    out = load_if_file(out)
    otherout = load_if_file(otherout)
    n = len(ref)
    assert n == len(out), 'Mismatched reference and output file size'
    assert n == len(otherout), 'Mismatched reference and other output file size'
    k = int(sample_percent * n / 100)
    bleus = []
    for i in range(num_samples):
        subset = np.random.choice(n, k)
        savetxt(dummy_out, out[subset])
        savetxt(dummy_otherout, otherout[subset])
        savetxt(dummy_ref, ref[subset])
        bleu1, _ = bleu_score(dummy_ref, dummy_out)
        bleu2, _ = bleu_score(dummy_ref, dummy_otherout)
        bleus.append([bleu1, bleu2])
        if verbose and (i + 1) % (num_samples // 10) == 0:
            print('%d%% done' % ((i + 1) // (num_samples // 10) * 10))
            sys.stdout.flush()
    bleus = np.asarray(bleus)
    return bleus


if __name__ == '__main__':
    args = parser.parse_args()
    ref = np.asarray(loadtxt(args.ref))
    out = np.asarray(loadtxt(args.out))
    n = len(ref)
    assert n == len(out), 'Mismatched reference and output file size'
    k = int(args.sample_size * n / 100)
    dummy_num = np.random.randint(1000000)
    dummy_out = '%d_out.txt' % dummy_num
    dummy_ref = '%d_ref.txt' % dummy_num
    if args.otherout is None:
        # Normal bootstrap resampling
        bleus = bootstrap_resampling(ref, out, args.num_samples,
                                     args.sample_size, dummy_ref,
                                     dummy_out, verbose=args.verbose)
        total, _ = bleu_score(args.ref, args.out)
        print('Total BLEU: %.3f' % total)
        print_stats(bleus)
        np.savetxt(args.bleufile, bleus)
    else:
        otherout = np.asarray(loadtxt(args.otherout))
        dummy_otherout = '%d_otherout.txt' % dummy_num
        bleus = bootstrap_resampling(ref, out, otherout,
                                     args.num_samples,
                                     args.sample_size,
                                     dummy_ref, dummy_out,
                                     dummy_otherout,
                                     verbose=args.verbose)
        print_paired_stats(bleus)
        np.savetxt(args.bleufile, bleus)
        os.remove(dummy_otherout)
    os.remove(dummy_out)
    os.remove(dummy_ref)
