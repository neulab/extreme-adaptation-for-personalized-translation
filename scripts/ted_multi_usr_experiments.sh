#!/bin/bash

LANG_PAIR=${1:-en_fr}
DATA_FOLDER=${2:-"ted/fr_en"}

#python make_lex.py -c config/ted_${LANG_PAIR}_base.yaml -e make_lex -en ted_${EXP_NAME}

for EXP_NAME in log_fact_voc
# fact_voc base fact_voc full_voc usr_token
do
    JOB_NAME=${EXP_NAME}_${LANG_PAIR} 
    sbatch -J $JOB_NAME scripts/run_medium_ted.sh ${JOB_NAME} config/ted_${LANG_PAIR}_${EXP_NAME}.yaml
done
