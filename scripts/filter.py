from __future__ import print_function, division
import os, sys, codecs
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Input file')
parser.add_argument('output_file', help='Output file')
parser.add_argument('--min_tok', '-m', type=int, default=1, help='Minimum sentence length')
parser.add_argument('--max_tok', '-M', type=int, default=60, help='Maximum sentence length')
parser.add_argument('--verbose', '-v', action='store_true', help='Verbosity')

class Logger(object):
    """Logging made easy"""
    def __init__(self, verbose=False):
        self.verbose = verbose

    def info(self, string):
        if self.verbose:
            print(string)
        sys.stdout.flush()

def load_tsv(filename, logger=Logger(False)):
    tsv=[]
    logger.info('Loading from tsv file %s' % filename)
    with codecs.open(filename, 'r', 'utf-8') as f:
        talks, src, trg = f.readline().strip().split('\t')
        logger.info('Headers: %s\t%s\t%s' % (talks, src, trg))
        for l in f:
            usr, src_s, trg_s = l.split('\t')
            tsv.append([usr, src_s, trg_s])
    return tsv

def save_tsv(filename, tsv):
    with codecs.open(filename, 'w+', 'utf-8') as f:
        for l in tsv:
            print('\t'.join(l), file=f)

def clean_up(dirty, min_tok=1, max_tok=60, logger=Logger(False)):
    nlong = nnull = nshort = 0
    N = len(dirty)
    clean = []
    for tlk, src_s, trg_s in dirty:
        src, trg = src_s.strip().split(), trg_s.strip().split()
        # If one of the sentences has a null, ignore (untranslated)
        if 'null' in trg_s.lower() or 'null' in src_s.lower():
            nnull+=1
            continue
        # Ignore sentences that are too long
        if len(src) > max_tok or len(trg) > max_tok:
            nlong+=1
            continue
        # Ignore sentences that are too short
        if len(src) < min_tok or len(trg) < min_tok:
            nshort+=1
            continue
        # Otherwise keep sentence pair
        clean.append([tlk.strip(), src_s.strip(), trg_s.strip()])
    # Print some info
    logger.info('Removed because too short: %d (%.1f%%)' % (nshort, nshort * 100 / N))
    logger.info('Removed because too long: %d (%.1f%%)' % (nlong, nlong * 100 / N))
    logger.info('Removed because contained null: %d (%.1f%%)' % (nnull, nnull * 100 / N))
    # Return clean data
    return clean

if __name__ == '__main__':
    args = parser.parse_args()
    logger = Logger(args.verbose)
    dirty = load_tsv(args.input_file, logger=logger)
    clean = clean_up(dirty, min_tok=args.min_tok, max_tok=args.max_tok, logger=logger)
    save_tsv(args.output_file, clean)


