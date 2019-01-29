from num2words import num2words as nw
import re
import pandas as pd
import numpy as np
from collections import Counter
import os

num_samples = 1000
num_upper_range = 10000


def gen_vocab():
    vocab = Counter(' '.join([num_to_word(x) for x in range(num_upper_range)]).split())
    vocab = pd.DataFrame(((k, v) for k, v in vocab.items()), columns=['tok', 'occ'])
    vocab.sort_values('occ', ascending=False, inplace=True)
    return vocab


def num_to_word(num):
    return re.sub(r'\W+', ' ', nw(num))


def gen_number_word_forms():
    X = np.reshape(np.random.randint(low=0, high=num_upper_range, size=num_samples), (-1, 1))
    X = np.concatenate([X, X + 1, X + 2], axis=1)
    X_str = [[num_to_word(el) for el in els] for els in list(X)]
    return X_str


def split_valid_train(X_str):
    train_size = int(0.9 * num_samples)
    X_str = np.array(X_str)
    X_train = X_str[:train_size]
    X_valid = X_str[train_size:]
    return X_train, X_valid


def gen_and_save(out_dir='data/nwords'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    vocab = gen_vocab()
    vocab.to_csv(os.path.join(out_dir, 'vocab.txt'), index=False, header=False, sep='\t')

    data = gen_number_word_forms()
    X_train, X_valid = split_valid_train(data)

    np.savetxt(fname=os.path.join(out_dir, 'train.txt'), X=X_train, delimiter='\t', fmt='%s')
    np.savetxt(fname=os.path.join(out_dir, 'valid.txt'), X=X_valid, delimiter='\t', fmt='%s')


if __name__ == '__main__':
    gen_and_save()
