from collections import Counter, defaultdict

class Tokenizer(object):
    def __init__(self,num_words=None,oov_char=1,split_by=' ',filters='#$%&()*+,-./:;<=>?@[\]^_`{|}'):
        self.num_words = num_words
        self.oov_char = oov_char
        self.split_by = split_by
        self.filters = filters
        self.word2id = None
        self.id2word = None
    
    def _word_filter(self, tokens):
        return list(map(lambda token: ''.join(list(filter(lambda x: x not in self.filters, token))), tokens))
    
    def _split(self, tokens):
        return list(map(lambda x: x.split(self.split_by), tokens))
    
    def _dedup(self, tokens_splited):
        return tuple(set(sum(tokens_splited, [])))
    
    def _word_count(self, tokens_splited):
        counter = Counter()
        for i, token in enumerate(tokens_splited):
            counter.update(token)
        return dict(counter)
    
    def build_vocab(self, tokens_splited):
        counter_dict = self._word_count(tokens_splited)
        sorted_count = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
        if self.num_words is None:
            word2id = {counter[0]: i+self.oov_char+1 for i, counter in enumerate(sorted_count)}
            id2word = {i+self.oov_char+1: counter[0] for i, counter in enumerate(sorted_count)}
        else:
            assert self.num_words <= len(sorted_count)
            word2id = {sorted_count[i][0]: i+self.oov_char+1 for i in range(self.num_words)}
            id2word = {i+self.oov_char+1: sorted_count[i][0] for i in range(self.num_words)}
        word2id = defaultdict(lambda: self.oov_char, word2id)
        id2word = defaultdict(lambda: '', id2word)
        self.word2id, self.id2word = word2id, id2word
    
    def map_token2id(self, *args):
        assert self.word2id is not None
        if len(args) == 1:
            tokens_splited = args[0]
            return list(map(lambda token: list(map(lambda x: self.word2id[x], token)), tokens_splited))
        else:
            output_list = []
            for tokens_splited in args:
                output_list.append(list(map(lambda token: list(map(lambda x: self.word2id[x], token)), tokens_splited)))
            return tuple(output_list)
    
    def map_id2token(self, id_list):
        assert self.id2word is not None
        return list(map(lambda sequence: list(map(lambda x: self.id2word[x], sequence)), id_list))
    
    def fit_text(self, *args):
        if len(args) != 1:
            tokens_all = sum(args,[])
        else:
            tokens_all = args[0]
        tokens_splited = self._split(self._word_filter(tokens_all))
        self.build_vocab(tokens_splited)
    
    def transform_text(self, *args):
        return self.map_token2id(*args)
    
    def get_vocab(self):
        assert self.word2id is not None and self.id2word is not None
        return self.word2id, self.id2word
    







