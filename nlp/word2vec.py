import argparse
import itertools
from pathlib import Path

import torch
from torch import nn, optim
from vecto.corpus import DirCorpus, tokenization
from vecto.vocabulary import Vocabulary

import wandb


def get_args():
    default_corpus = Path("../data/nlp/corpora/BNC")
    default_vocab = default_corpus / "m10"

    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb-project", default="simple word2vec")
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
    def __init__(self, args, corpus: DirCorpus, vocab: Vocabulary):
        super().__init__()
        self.corpus = corpus
        self.vocab = vocab
        self.emb = nn.Embedding(vocab.cnt_words, args.embed_width)
        nn.init.xavier_normal_(self.emb.weight)
        self.linear = nn.Linear(args.embed_width, vocab.cnt_words)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, batch):
        h = self.emb(batch)
        return self.linear(h)


def evaluate(net: SimpleWord2Vec, vocab: Vocabulary):
    vemb = WordEmbeddingsDense()
    vemb.matrix = net.emb.weight.detach().numpy()
    word = vocab.get_frequency(1)
    similar = vemb.get_most_similar_words(word)
    return word, similar


def train(batch, device, net, optimizer, criterion):
    inputs = batch[0].to(device)
    target = batch[1].to(device)
    optimizer.zero_grad()
    logits = net(inputs)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def main(args):
    wandb.init(project=args.wandb_project)
    wandb.config.update(args)

    print("start word2vec")

    corpus = DirCorpus(args.corpus_path)
    vocab = Vocabulary()
    vocab.load(args.vocab_path / "normal")
    sliding_windows = corpus.get_sliding_window_iterator()
    dataset = list(expand_sliding_windows(sliding_windows, vocab.get_id))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SimpleWord2Vec(args, corpus, vocab)
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(dataset),
    )

    count = 0
    for epoch in range(args.epochs):
        print("EPOCH:", epoch)
        corpus = DirCorpus(args.corpus_path)
        for batch in BatchIter(dataset, args.batch_size):
            loss = train(batch, device, net, optimizer, criterion)
            scheduler.step()
            wandb.log({"loss": loss.item(), "lr": scheduler.get_lr()})
            if count & 127 == 0:
                print(f"step: {count:6}, " f"loss {loss:6.2f}, ")
                torch.save(net.state_dict(), args.save_path)

            count += 1


def altmain():
    emb = nn.Embedding(3, 4)
    print(emb.weight.detach().numpy())


if __name__ == "__main__":
    args = get_args()
    # altmain()
    main(args)
