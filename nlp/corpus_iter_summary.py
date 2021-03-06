import vecto.corpus


def head(get_iter, n=10):
    count = 0
    for itr in get_iter:
        print(itr)
        count += 1
        if count == n:
            break


def section(path, fun, args=[]):
    expanded = ", ".join(map(str, args))
    print(f"# {fun}({expanded})")
    corpus = vecto.corpus.DirCorpus(path)
    get_iter = eval(f"corpus.{fun}")
    print("```")
    head(get_iter(*args))
    print("```")
    print()


def main():
    corpora_path = "../data/nlp/corpora"
    tokenizer = vecto.corpus.tokenization.DEFAULT_TOKENIZER
    sequence_length = 4

    section(corpora_path, "get_token_iterator", args=[tokenizer])
    section(corpora_path, "get_line_iterator")
    section(corpora_path, "get_sentence_iterator", args=[tokenizer])
    section(corpora_path, "get_sequence_iterator", args=[sequence_length, tokenizer])
    section(corpora_path, "get_sliding_window_iterator", args=[2, 3, tokenizer])


if __name__ == "__main__":
    main()
