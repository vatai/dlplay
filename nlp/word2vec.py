import argparse
from pathlib import Path

from vecto.corpus import DirCorpus, tokenization
from vecto.vocabulary import Vocabulary

# import torch


# def mkmodel():
#     pass


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("../data/nlp/corpora/BNC"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
