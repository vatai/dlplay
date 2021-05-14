import argparse
import itertools
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from vecto.corpus import DirCorpus, tokenization
from vecto.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("../data/nlp/corpora/BNC"))
    parser.add_argument("--embed-width", type=int, default=512)
    return parser.parse_args()


class BatchIter:
    def __init__(self, outs, n, collect_fn=list):
        self.outs = outs
        self.n = n
        self.collect = collect_fn

    def __next__(self):
        result = self.collect(itertools.islice(self.outs, self.n))
        if result != []:
            return result
        else:
            raise StopIteration

    def __iter__(self):
        return self


def expand_sliding_windows(itr):
    outs = map(lambda t: itertools.product(t["ctx"], [t["cur"]]), itr)
    return itertools.chain.from_iterable(outs)


class SimpleWord2Vec(nn.Module):
    def __init__(self, args, corpus, vocab):
        super().__init__()
        self.corpus = corpus
        self.vocab = vocab
        self.emb = nn.Embedding(vocab.cnt_words, args.embed_width)
        self.linear = nn.Linear(args.embed_width, vocab.cnt_words)

    def forward(self, batch):
        h = self.emb(batch)
        h = self.linear(h)
        return F.relu(h)


def main(args):
    print("start word2vec")
    corpus = DirCorpus(args.path)
    vocab = Vocabulary()
    tokenizer = tokenization.DEFAULT_TOKENIZER

    vocab.load(args.path / "m10/normal")
    model = SimpleWord2Vec(args, corpus, vocab)
    model.train()

    count = 0
    for itr in corpus.get_sliding_window_iterator(tokenizer=tokenizer):
        cur, ctx = itr["current"], itr["context"]
        cur_id = vocab.get_id(cur)
        ctx_ids = list(map(vocab.get_id, ctx))
        # print(cur, ctx)
        # print(cur_id, list(ctx_ids))
        count += 1

        if len(ctx_ids) > 0:
            outs = model(torch.LongTensor([[cur_id, ctx_ids[0]]]))
            print("!!!!!!!!!!", outs)
        if count == 10:
            break


def altmain():
    print()

    d = [{"cur": "a", "ctx": [1, 2, 3]}, {"cur": "b", "ctx": [5, 6, 7]}]
    outs = map(lambda t: itertools.product(t["ctx"], [t["cur"]]), d)
    outs = itertools.chain.from_iterable(outs)

    count = 0
    for batch in BatchIter(outs, 4):
        print(batch)
        count += 1
        if count == 10:
            break


if __name__ == "__main__":
    args = get_args()
    main(args)
    # altmain()
