#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=512M
#SBATCH -t 0
#REF=ted_fr-en_tlk/test.multi.en
REF=${1:-../multi-speak/ted_es/mono/test.en}
#OUT=../multi-speak/output/base_test.out
OUT=${2:-../multi-speak/output/base_es_en_mono_test.out}
OTHEROUT=${3:-../multi-speak/output/wavgwv_user_init_es_en_mono_test.out}
LOG=${4:-bleus.txt}
python evaluation.py $REF $OUT $OTHEROUT -M 1000 -v --bleufile bleus_base_user_init.txt > $LOG
