import vecto

from util import head


def main():
    path = "../data/nlp"
    corpus = vecto.corpus.DirCorpus(path)
    print("hello")

    head(corpus.get_line_iterator)


if __name__ == "__main__":
    main()
