#!/bin/bash

set_env /work/opt/cuda/nccl_2.7.8-1+cuda10.2_x86_64/
set_env /work/opt/cuda/cudnn-10.0-linux-x64-v7.4.2.24/
spack load gcc@8.4.0

python -m \
	vecto.embeddings.train_word2vec \
	--gpu=0 \
        --window=5 \
	--epoch=10 \
	--path_out=w2v \
	--path_corpus=../BNC/AA/ \
        --path_vocab=../BNC/m100/normal

# From alex
# epoch=4
# python3 -m vecto.embeddings.train_word2vec \
#     --path_vocab=/work/data/NLP/vocabs/brown_m5 \
#     --path_corpus=/work/data/NLP/corpora/raw_texts/Eng/BNC \
