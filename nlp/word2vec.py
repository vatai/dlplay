import argparse
import datetime
import itertools
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset
from vecto.corpus import DirCorpus
from vecto.vocabulary import Vocabulary

import wandb


def get_utc_timestamp():
    utcnow = datetime.datetime.utcnow()
    return utcnow.strftime("%Y%m%d_%H%M%S_UTC")


def get_args():
    default_corpus = Path("../data/nlp/corpora/BNC")
    default_vocab = default_corpus / "m10"
    default_lr = 0.0001

    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb-project", default="simple word2vec")
    parser.add_argument("--corpus-path", type=Path, default=default_corpus)
    parser.add_argument("--vocab-path", type=Path, default=default_vocab)
    parser.add_argument("--save-path", type=Path, default=Path("./last.chkp"))
    parser.add_argument("--embed-width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="SGD")

    # SGD
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--momentum", type=float, default=0.00)
    # Adam
    # AdamW

    # OneCycleLR
    parser.add_argument("--max-lr", type=float, default=default_lr)
    parser.add_argument("--steps-per-epoch", type=int, default=500000)
    parser.add_argument("--div-factor", type=float, default=1.0)
    parser.add_argument("--pct-start", type=float, default=0.0)

    return parser.parse_args()


class NgramDataset(IterableDataset):
    def __init__(self, corpus: DirCorpus, vocab: Vocabulary):
        self.corpus = corpus
        self.vocab = vocab

    def _get_context_current_product(self, wnd_dict):
        cxt_ids = filter(bool, map(self.vocab.get_id, wnd_dict["context"]))
        cur_ids = filter(bool, [self.vocab.get_id(wnd_dict["current"])])
        return itertools.product(cur_ids, cxt_ids)

    def __iter__(self):
        sw_dicts = self.corpus.get_sliding_window_iterator()
        ctx_cur_prods = map(self._get_context_current_product, sw_dicts)
        self.iterator = itertools.chain.from_iterable(ctx_cur_prods)
        return self.iterator

    def __next__(self):
        return next(self.iterator)


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
    dataset = NgramDataset(corpus, vocab)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SimpleWord2Vec(args, corpus, vocab)
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    else:
        raise ValueError("Wrong --optim")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        div_factor=args.div_factor,
        pct_start=args.pct_start,
        anneal_strategy="linear",
    )

    step = 0
    for epoch in range(args.epochs):
        print("EPOCH:", epoch)
        for batch in DataLoader(dataset, args.batch_size):
            loss = train(batch, device, net, optimizer, criterion)
            scheduler.step()
            last_lr = scheduler.get_last_lr()[0]

            if step & ((2 ** 12) - 1) == 0:
                wandb.log({"step": step, "loss": loss, "lr": last_lr})
                print(f"step: {step:6}, loss {loss:6.2f}, lr: {last_lr}")
            step += 1
        torch.save(net.state_dict(), args.save_path)

    save_weights_numpy(net)


def save_weights_numpy(net):
    weights = net.cpu().emb.weight.detach().numpy()
    with open(str(args.save_path) + ".npy", "wb") as file:
        np.save(file, weights)


def load_main(args):
    print("loading")
    corpus = DirCorpus(args.corpus_path)
    vocab = Vocabulary()
    vocab.load(args.vocab_path / "normal")
    net = SimpleWord2Vec(args, corpus, vocab)
    net.load_state_dict(torch.load(args.save_path))
    print(net)
    save_weights_numpy(net)


def altmain():
    emb = nn.Embedding(3, 4)
    print(emb.weight.detach().numpy())


if __name__ == "__main__":
    args = get_args()
    # altmain()
    main(args)
