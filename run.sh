#!/bin/sh
HYDRA_FULL_ERROR=1 python train.py +experiments=dna/cherimoya logger.do.online=True
# python gflownet/proxy/sequences/cherimoya/seqs/find_best_section.py
# HYDRA_FULL_ERROR=1 python eval.py rundir=/home/mcb/users/ehuang14/classes/gflownets/gflownet/logs/dna_cherimoya/2026-11-26_08-43-56_377087 n_samples=1000
# python gflownet/proxy/sequences/cherimoya/seqs/plot_best_seqs.py

GREEN='\033[1;32m'
NC='\033[0m' # No Color
printf "${GREEN}FINISHED${NC}\n"
