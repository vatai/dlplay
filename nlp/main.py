import vecto.vocabulary


def main():
    v = vecto.vocabulary.Vocabulary()

    path = "../data/nlp/corpora/BNC/m10/normal"
    v.load(path)

    print(v.lst_words[:10])

    for word in ["one", "said", "like"]:
        idx = v.get_id(word)
        print(f"index of {word}: {idx}")

    word = "megszentsegtelenethetetleneskedeseitekert"
    idx = v.get_id(word)
    print(f"index of {word}: {idx}")
    print(len(word))


if __name__ == "__main__":
    main()
