import vecto.vocabulary


def pprint(word, vocab):
    idx = vocab.get_id(word)
    print(f"index of {word}: {idx}")


def main():
    v = vecto.vocabulary.Vocabulary()

    path = "../data/nlp/corpora/BNC/m10/normal"
    v.load(path)

    print(v.lst_words[:10])

    for word in ["one", "said", "like"]:
        pprint(word, v)

    pprint("megszentsegtelenethetetleneskedeseitekert", v)
    pprint("[UNK]", v)


if __name__ == "__main__":
    main()
