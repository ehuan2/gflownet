#!/bin/sh
# HYDRA_FULL_ERROR=1 python train.py +experiments=dna/cherimoya logger.do.online=True
python gflownet/proxy/sequences/cherimoya/seqs/find_best_section.py

GREEN='\033[1;32m'
NC='\033[0m' # No Color
printf "${GREEN}FINISHED${NC}\n"
