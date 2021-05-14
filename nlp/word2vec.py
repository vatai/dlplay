import argparse
import functools
import itertools
from pathlib import Path

import torch
from torch import nn, optim
from vecto.corpus import DirCorpus, tokenization
from vecto.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("../data/nlp/corpora/BNC"))
    parser.add_argument("--embed-width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    return parser.parse_args()


class BatchIter:
    def __init__(self, outs, n, collect_fn=list):
        self.outs = outs
        self.n = n
        self.collect = collect_fn

    def __next__(self):
        slice = itertools.islice(self.outs, self.n)
        result = self.collect(slice)
        if result != []:
            batch = tuple(zip(*result))
            return list(map(torch.LongTensor, batch))
        else:
            raise StopIteration

    def __iter__(self):
        return self


def expand_sliding_windows(itr, fn=lambda t: t):
    outs = map(
        lambda t: itertools.product(
            map(fn, t["context"]),
            [fn(t["current"])],
        ),
        itr,
    )
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
        return self.linear(h)


def main(args):
    print("start word2vec")
    corpus = DirCorpus(args.path)
    vocab = Vocabulary()
    vocab.load(args.path / "m10/normal")
    tokenizer = tokenization.DEFAULT_TOKENIZER

    net = SimpleWord2Vec(args, corpus, vocab)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    debug = False
    for epoch in range(args.epochs):
        count = 0
        print("EPOCH:", epoch)
        corpus = DirCorpus(args.path)
        sliding_windows = corpus.get_sliding_window_iterator(tokenizer=tokenizer)
        expanded_iter = expand_sliding_windows(sliding_windows, vocab.get_id)
        for inputs, target in BatchIter(expanded_iter, args.batch_size):

            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            if count & 31 == 0:
                print(f"step: {count}, loss {loss.item()}")

            if debug:
                if count < 3:
                    print("INPUTS:", inputs)
                    print("TARGET:", target)
                    # print("---")
                    print("LOSS:", loss.item())
                else:
                    break
            count += 1


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
