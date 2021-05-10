import vecto.vocabulary


def main():
    v = vecto.vocabulary.Vocabulary()

    path = "../data/nlp/vocabs/bnc/m10/"
    v.load(path)

    print(v.lst_words[:10])

    for word in ["one", "said", "like"]:
        idx = v.get_id(word)
        print(f"index of {word}: {idx}")


if __name__ == "__main__":
    main()
