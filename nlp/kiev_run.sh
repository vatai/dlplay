# git pull
# python word2vec.py --corpus-path ../enwiki_2021.01/AA --vocab-path ../enwiki_2021.01/m100
# python word2vec.py --corpus-path ../BNC/AA --vocab-path ../BNC/m100
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.01  --max-lr=0.01
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.005 --max-lr=0.005
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.001
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.001 --momentum=0.5
# python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.005 --momentum=0.5

python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.05 --pct-start=0.1 --momentum=0.5

python word2vec.py --corpus-path=../BNC/AA --vocab-path ../BNC/m100 --lr=0.001 --max-lr=0.05 --pct-start=0.1 --momentum=0.5 --steps-per-epoch=50000
