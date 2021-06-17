#!/bin/bash

# . ~/.bashrc
# set_env /work/opt/cuda/nccl_2.7.8-1+cuda10.2_x86_64/
# set_env /work/opt/cuda/cudnn-10.0-linux-x64-v7.4.2.24/

# git pull
# python word2vec.py --corpus-path ../enwiki_2021.01/AA --vocab-path ../enwiki_2021.01/m100
# python word2vec.py --corpus-path ../BNC/AA --vocab-path ../BNC/m100
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.01  --max-lr=0.01
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.005 --max-lr=0.005
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.001
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.001 --momentum=0.5
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.005 --momentum=0.5

# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.05 --pct-start=0.1 --momentum=0.5

# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.05 --pct-start=0.1 --momentum=0.5 --steps-per-epoch=50000

# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.05 --pct-start=0.1 --momentum=0.5 --steps-per-epoch=50000 --div-factor=25

python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.05 --pct-start=0.1 --momentum=0.5 --steps-per-epoch=2000000 --div-factor=25 --embed-width=300

mv last.chkp.npy tmp/normal/

rm -rf my_analogy/*

python -m vecto benchmark analogy --method LRCos --path_out=my_analogy tmp/normal/ BATS_3.0

./analogy_plot.py my_analogy
