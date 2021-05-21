#!/bin/bash

set_env /work/opt/cuda/nccl_2.7.8-1+cuda10.2_x86_64/
set_env /work/opt/cuda/cudnn-10.0-linux-x64-v7.4.2.24/
spack load gcc@8.4.0
python -m vecto.embeddings.train_word2vec --gpu=0 --path_corpus=../BNC/AA/ --path_out=vecto_w2v
