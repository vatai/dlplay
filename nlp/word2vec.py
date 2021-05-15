import argparse
import functools
import itertools
from pathlib import Path

import torch
import wandb
from torch import nn, optim
from vecto.corpus import DirCorpus, tokenization
from vecto.vocabulary import Vocabulary


def get_args():
    default_corpus = Path("../data/nlp/corpora/BNC")
    default_vocab = default_corpus / "m10"

    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus-path", type=Path, default=default_corpus)
    parser.add_argument("--vocab-path", type=Path, default=default_vocab)
    parser.add_argument("--save-path", type=Path, default=Path("./last.chkp"))
    parser.add_argument("--embed-width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.99)

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
        nn.init.xavier_normal_(self.emb.weight)
        self.linear = nn.Linear(args.embed_width, vocab.cnt_words)
        nn.init.xavier_normal_(self.linear.weight)
        # nn.init.xavier_normal_(self.linear.bias)

    def forward(self, batch):
        h = self.emb(batch)
        return self.linear(h)


def main(args):
    wandb.init(project=__file__)
    wandb.config.update(args)

    print("start word2vec")
    corpus = DirCorpus(args.corpus_path)
    vocab = Vocabulary()
    vocab.load(args.corpus_path / "normal")
    tokenizer = tokenization.DEFAULT_TOKENIZER

    net = SimpleWord2Vec(args, corpus, vocab)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    debug = False
    cumulative_loss = 0.0
    count = 0
    net.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    for epoch in range(args.epochs):
        print("EPOCH:", epoch)
        corpus = DirCorpus(args.corpus_path)
        sliding_windows = corpus.get_sliding_window_iterator(tokenizer=tokenizer)
        expanded_iter = expand_sliding_windows(sliding_windows, vocab.get_id)
        for inputs, target in BatchIter(expanded_iter, args.batch_size):

            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, target)
            loss.backward()
            cumulative_loss += loss.item()
            running_loss = cumulative_loss / (count + 1)
            optimizer.step()

            if count & 127 == 0:
                print(
                    f"step: {count:6}, "
                    f"loss {loss.item():6.2f}, "
                    f"running loss: {(running_loss):6.2f}"
                )
                torch.save(net.state_dict(), args.save_path)

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
    step, loss = 32, 3.141592
    print(f"step: {step:6}, loss: {loss:5.2f}")


if __name__ == "__main__":
    args = get_args()
    main(args)
    # altmain()
