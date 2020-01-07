from os.path import join
import pickle as pkl
from collections import Counter
import argparse

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


class Vocab(object):
    def __init__(self):
        self.token2idx = {"<s>": START_TOKEN, "</s>": END_TOKEN, "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
        self.idx2token = dict((idx, token) for token, idx in self.token2idx.items())
        self.length = 4

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.length
            self.idx2token[self.length] = token
            self.length += 1

    def __len__(self):
        return self.length


def build_vocab(data_dir, min_count=10):
    vocab = Vocab()
    counter = Counter()

    formulas_file = join(data_dir, 'im2latex_formulas.norm.lst')
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    with open(join(data_dir, 'im2latex_train_filter.lst'), 'r') as f:
        for line in f:
            _, idx = line.strip('\n').split()
            idx = int(idx)
            formula = formulas[idx].split()
            counter.update(formula)
    # new = 'train_vocab.txt'
    # f_new = open(new, 'w')
    for word, count in counter.most_common():
        if count >= min_count:
            vocab.add_token(word)
            # print(word)
    #         f_new.write(word+'\n')
    # f_new.close()
    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing Vocab File in ", vocab_file)
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)


def load_vocab(data_dir):
    with open(join(data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
    print("Load vocab including {} words!".format(len(vocab)))
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building vocab for Im2Latex")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    args = parser.parse_args()
    vocab = build_vocab(args.data_path)
    # load_vocab(args.data_path)