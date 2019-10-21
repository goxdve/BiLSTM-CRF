import random
import torch


def read_corpus(filepath):
    """ Read corpus from the given file path.
    Args:
        filepath: file path of the corpus
    Returns:
        sentences: a list of sentences, each sentence is a list of str
        tags: corresponding tags
    """
    sentences, tags = [], []
    sent, tag = ['<START>'], ['<START>']
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            if line == '\n':
                if len(sent) > 1:
                    sentences.append(sent + ['<END>'])
                    tags.append(tag + ['<END>'])
                sent, tag = ['<START>'], ['<START>']
            else:
                line = line.split()
                sent.append(line[0])
                tag.append(line[1])
    return sentences, tags


def generate_train_dev_dataset(filepath, sent_vocab, tag_vocab, train_proportion=0.8):
    """ Read corpus from given file path and split it into train and dev parts
    Args:
        filepath: file path
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        train_proportion: proportion of training data
    Returns:
        train_data: data for training, list of tuples, each containing a sentence and corresponding tag.
        dev_data: data for development, list of tuples, each containing a sentence and corresponding tag.
    """
    sentences, tags = read_corpus(filepath)
    sentences = words2indices(sentences, sent_vocab)
    tags = words2indices(tags, tag_vocab)
    data = list(zip(sentences, tags))
    random.shuffle(data)
    n_train = int(len(data) * train_proportion)
    train_data, dev_data = data[: n_train], data[n_train:]
    return train_data, dev_data


def batch_iter(data, batch_size=32, shuffle=True):
    """ Yield batch of (sent, tag), by the reversed order of source length.
    Args:
        data: list of tuples, each tuple contains a sentence and corresponding tag.
        batch_size: batch size
        shuffle: bool value, whether to random shuffle the data
    """
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags = [x[1] for x in batch]
        yield sentences, tags


def words2indices(origin, vocab):
    """ Transform a sentence or a list of sentences from str to int
    Args:
        origin: a sentence of type list[str], or a list of sentences of type list[list[str]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with int
    """
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result


def indices2words(origin, vocab):
    """ Transform a sentence or a list of sentences from int to str
    Args:
        origin: a sentence of type list[int], or a list of sentences of type list[list[int]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with str
    """
    if isinstance(origin[0], list):
        result = [[vocab.id2word(w) for w in sent] for sent in origin]
    else:
        result = [vocab.id2word(w) for w in origin]
    return result


def pad(data, padded_token, device):
    """ pad data so that each sentence has the same length as the longest sentence
    Args:
        data: list of sentences, List[List[word]]
        padded_token: padded token
        device: device to store data
    Returns:
        padded_data: padded data, a tensor of shape (max_len, b)
        lengths: lengths of batches, a list of length b.
    """
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


def print_var(**kwargs):
    for k, v in kwargs.items():
        print(k, v)


def main():
    sentences, tags = read_corpus('data/train.txt')
    print(len(sentences), len(tags))


if __name__ == '__main__':
    main()
