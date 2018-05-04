#!/bin/bash

function get_ppl {
    grep -E "Dev.*ppl" $1 | awk -F" |,|=" '{print NR " " $8}' > $2
}

function get_bleu {
    grep "BLEU =" $1 | awk -F" |,"  '{print NR " " $3}' > $2
}

NAME=$1
FILE1=$2
FILE2=$3
DATA_FILE1=temp/datafile1
DATA_FILE2=temp/datafile2
NAME1=$(basename $FILE1 .txt)
NAME2=$(basename $FILE2 .txt)
get_bleu $FILE1 $DATA_FILE1
get_bleu $FILE2 $DATA_FILE2
GNUPLOT_CMD="set terminal png;
set output '${NAME}.png';
set style line;
set datafile separator ' ';
plot '${DATA_FILE1}' using 1:2 title '${NAME1}', 
'${DATA_FILE2}' using 1:2 title '${NAME2}';"
echo -e $GNUPLOT_CMD | gnuplot -p

