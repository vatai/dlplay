import argparse
from pathlib import Path

from torch import nn
from vecto.corpus import DirCorpus, tokenization
from vecto.vocabulary import Vocabulary


def get_model(args, cnt_words):
    embeddings = nn.Embedding(cnt_words, args.embed_width)
    return embeddings


def train(model, corpus, vocab):
    pass


def main(args):
    print("start word2vec")
    corpus = DirCorpus(args.path)
    vocab = Vocabulary()
    tokenizer = tokenization.DEFAULT_TOKENIZER

    vocab.load(args.path / "m10/normal")
    count = 0
    for itr in corpus.get_sliding_window_iterator(tokenizer=tokenizer):
        cur, ctx = itr["current"], itr["context"]
        cur_id = vocab.get_id(cur)
        ctx_ids = map(vocab.get_id, ctx)
        print(cur, ctx)
        print(cur_id, list(ctx_ids))
        count += 1
        if count == 10:
            break

    model = get_model(args, vocab.cnt_words)
    train(model, corpus, vocab)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("../data/nlp/corpora/BNC"))
    parser.add_argument("--embed-width", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
