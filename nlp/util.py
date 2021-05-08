def head(get_iter, n=10):
    count = 0
    for itr in get_iter():
        print(itr)
        count += 1
        if count == n:
            return
