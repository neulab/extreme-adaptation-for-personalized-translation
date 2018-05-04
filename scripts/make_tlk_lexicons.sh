#!/bin/bash
#SBATCH --nodelist=compute-0-7
#SBATCH -n 20
#SBATCH -o log_make_tlk_lexicons.txt
DIR=$1
LANG_SRC=$2
LANG_TRG=$3
i=0
for f_src in ${DIR}/*.${LANG_SRC}; do
    f_trg=${f_src%.$LANG_SRC}.$LANG_TRG
    f_name=`basename $f_trg .$LANG_TRG`
    f_lex=${f_src%.$LANG_SRC}.lex
    ((i++))
    if ! ((i % 10)); then
        bash scripts/make_lexicon.sh $f_src $f_trg $f_lex
    else
        bash scripts/make_lexicon.sh $f_src $f_trg $f_lex &
    fi
done

