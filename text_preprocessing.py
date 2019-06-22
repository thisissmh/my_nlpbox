from collections import Counter, defaultdict


def word_filter(tokens, filters='#$%&()*+,-./:;<=>?@[\]^_`{|}'):
    return list(map(lambda token: ''.join(list(filter(lambda x: x not in filters, token))), tokens))


def split(tokens, by=' '):
    """
    param tokens: a list of string.
    """
    return list(map(lambda x: x.split(by), tokens))


def dedup(tokens_splited):
    """
    param tokens_splited: a list of word list, which is split from token
    """
    return tuple(set(sum(tokens_splited, [])))


def word_count(tokens_splited):
    """
    param tokens_splited: a list of word list, which is split from token
    """
    counter = Counter()
    for i, token in enumerate(tokens_splited):
        counter.update(token)
    return dict(counter)


def build_vocab(tokens_splited, num_words=None, oov_char=1):
    counter_dict = word_count(tokens_splited)
    sorted_count = sorted(counter_dict.items(),
                          key=lambda x: x[1], reverse=True)
    if num_words is None:
        word2id = {counter[0]: i+oov_char+1 for i, counter in enumerate(sorted_count)}
        id2word = {i+oov_char+1: counter[0] for i, counter in enumerate(sorted_count)}
    else:
        assert num_words <= len(sorted_count)
        word2id = {sorted_count[i][0]: i+oov_char+1 for i in range(num_words)}
        id2word = {i+oov_char+1: sorted_count[i][0] for i in range(num_words)}
    word2id = defaultdict(lambda: oov_char, word2id)
    id2word = defaultdict(lambda: '', id2word)
    return word2id, id2word


def map_token2id(word2id, *args):
    if len(args) == 1:
        tokens_splited = args[0]
        return list(map(lambda token: list(map(lambda x: word2id[x], token)), tokens_splited))
    else:
        output_list = []
        for tokens_splited in args:
            output_list.append(
                list(map(lambda token: list(map(lambda x: word2id[x], token)), tokens_splited)))
        return tuple(output_list)


def map_id2token(id_list, id2word):
    return list(map(lambda sequence: list(map(lambda x: id2word[x], sequence)), id_list))
