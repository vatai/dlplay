import vecto.vocabulary


def main():
    v = vecto.vocabulary.Vocabulary()

    path = "../data/nlp/BNC/bnc.txt"
    v.load_from_list(path)

    print(v.lst_words)


if __name__ == "__main__":
    main()
