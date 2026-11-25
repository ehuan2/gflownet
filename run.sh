#!/bin/sh
HYDRA_FULL_ERROR=1 python train.py +experiments=dna/cherimoya logger.do.online=True

GREEN='\033[1;32m'
NC='\033[0m' # No Color
printf "${GREEN}FINISHED${NC}\n"
