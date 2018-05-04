#!/bin/bash
#SBATCH --nodelist=compute-0-7
SRC_FILE=$1
TRG_FILE=$2
OUT_FILE=$3

FAST_ALIGN=fast_align
ID=`uuidgen`
TEMP_DATA_FILE=aligned-$ID
TEMP_ALIGN_FILE=alignements-$ID
SCRIPT_DIR=~/wd/multi-speak/scripts
paste $SRC_FILE $TRG_FILE | awk -F "\t" '{printf "%s ||| %s\n", $1, $2}'> $TEMP_DATA_FILE
$FAST_ALIGN -i $TEMP_DATA_FILE -d -o -v > $TEMP_ALIGN_FILE

python $SCRIPT_DIR/align_to_lexicon.py $TEMP_DATA_FILE $TEMP_ALIGN_FILE $OUT_FILE

rm $TEMP_DATA_FILE $TEMP_ALIGN_FILE

