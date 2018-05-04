#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -t 0
EN=${1:-sample}
CFG_FILE=${2:-"config/medium_ted_config.yaml"}
TMP_CFG_FILE=temp/`uuidgen`.yaml
sed '/^#/ d' $CFG_FILE > $TMP_CFG_FILE
# Training
python run.py -c $TMP_CFG_FILE -e train --exp_name $EN > output/log_${EN}.txt 2>&1
# Testing
#python run.py -c $TMP_CFG_FILE -e test --exp_name $EN >> output/log_${EN}.txt 2>&1
# Tuning
python run.py -c $TMP_CFG_FILE -e tune --exp_name $EN >> output/log_${EN}.txt 2>&1
# Testing
python run.py -c $TMP_CFG_FILE -e test --exp_name $EN >> output/log_${EN}.txt 2>&1
# Adaptation
#python new_users.py -c $TMP_CFG_FILE -e eval --exp_name $EN --update_mode full > output/log_${EN}_adapt_full.txt 2>&1
# 
#python new_users.py -c $TMP_CFG_FILE -e eval --exp_name $EN --update_mode biases > output/log_${EN}_adapt_biases.txt 2>&1
#
#python new_users.py -c $TMP_CFG_FILE -e eval --exp_name $EN --update_mode mixture_weights > output/log_${EN}_adapt_mixture_weights.txt 2>&1

