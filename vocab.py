"""
Usage:
    vocab.py TRAIN SENT_VOCAB TAG_VOCAB [options]

Options:
    --max-size=<int>   maximum size of the dictionary [default: 5000]
    --freq-cutoff=<int>     frequency cutoff [default: 2]
"""
from itertools import chain
from collections import Counter
from utils import read_corpus
from docopt import docopt
import json


class Vocab:
    def __init__(self, word2id, id2word):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.__word2id = word2id
        self.__id2word = id2word

    def get_word2id(self):
        return self.__word2id

    def get_id2word(self):
        return self.__id2word

    def __getitem__(self, item):
        if self.UNK in self.__word2id:
            return self.__word2id.get(item, self.__word2id[self.UNK])
        return self.__word2id[item]

    def __len__(self):
        return len(self.__word2id)

    def id2word(self, idx):
        return self.__id2word[idx]

    @staticmethod
    def build(data, max_dict_size, freq_cutoff, is_tags):
        """ Build vocab from the given data
        Args:
            data (List[List[str]]): List of sentences, each sentence is a list of str
            max_dict_size (int): The maximum size of dict
                                 If the number of valid words exceeds dict_size, only the most frequently-occurred
                                 max_dict_size words will be kept.
            freq_cutoff (int): If a word occurs less than freq_size times, it will be dropped.
            is_tags (bool): whether this Vocab is for tags
        Returns:
            vocab: The Vocab instance generated from the given data
        """
        word_counts = Counter(chain(*data))
        valid_words = [w for w, d in word_counts.items() if d >= freq_cutoff]
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[: max_dict_size]
        valid_words += ['<PAD>']
        word2id = {w: idx for idx, w in enumerate(valid_words)}
        if not is_tags:
            word2id['<UNK>'] = len(word2id)
            valid_words += ['<UNK>']
        return Vocab(word2id=word2id, id2word=valid_words)

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump({'word2id': self.__word2id, 'id2word': self.__id2word}, f, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf8') as f:
            entry = json.load(f)
        return Vocab(word2id=entry['word2id'], id2word=entry['id2word'])


def main():
    args = docopt(__doc__)
    sentences, tags = read_corpus(args['TRAIN'])
    sent_vocab = Vocab.build(sentences, int(args['--max-size']), int(args['--freq-cutoff']), is_tags=False)
    tag_vocab = Vocab.build(tags, int(args['--max-size']), int(args['--freq-cutoff']), is_tags=True)
    sent_vocab.save(args['SENT_VOCAB'])
    tag_vocab.save(args['TAG_VOCAB'])


if __name__ == '__main__':
    main()
