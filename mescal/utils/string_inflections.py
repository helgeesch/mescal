def to_plural(word: str) -> str:
    if (
            word.endswith('s')
            or word.endswith('x')
            or word.endswith('z')
            or word.endswith('ch')
            or word.endswith('sh')
    ):
        return word + 'es'
    elif word.endswith('y') and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    else:
        return word + 's'


def to_singular(word: str) -> str:
    if word.endswith('ies') and len(word) > 3 and word[-4] not in 'aeiou':
        return word[:-3] + 'y'
    elif word.endswith('es'):
        if (
                word[:-2].endswith('s')
                or word[:-2].endswith('x')
                or word[:-2].endswith('z')
                or word[:-2].endswith('ch')
                or word[:-2].endswith('sh')
        ):
            return word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


if __name__ == '__main__':
    test_words = ['cat', 'bus', 'country', 'dog', 'box', 'church', 'baby', 'day']

    print("Singulars to plurals:")
    for w in test_words:
        print(f"{w} → {to_plural(w)}")

    plurals = [to_plural(word) for word in test_words]
    print("\nPlurals back to singulars:")
    for w in plurals:
        print(f"{w} → {to_singular(w)}")