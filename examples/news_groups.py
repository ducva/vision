import os
import sys

from fastai.imports.core import *
from fastai.text.data import TextLMDataBunch, TextClasDataBunch
from fastai.text.transform import Vocab

sys.path.append("../vision")
from vision.datasets import *
from vision.text.transform import TextTokenizer

URI = 'https://s3.amazonaws.com/datasart-ds/20_newsgroup.tgz'
TEXT_DATA_DIR = "./data/20_newsgroup"


def parse_text_data():
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)

    print('Found %s texts.' % len(texts))
    df: pd.DataFrame = pd.DataFrame.from_dict({'text': texts, 'label': labels})
    df.to_csv('./data/20_newsgroup.csv')
    return texts, labels, labels_index


def create_data_bunch():
    """
    A data bunch includes:
    - itos object
    - train_texts
    - train_labels
    - val_texts
    - val_labels
    - test_texts
    - test_labels
    :return:
    """
    pass


def tokenizer(texts):  # create a tokenizer function
    tok = TextTokenizer('en')
    return tok.process_all(texts)


if __name__ == "__main__":
    # 1. Download data
    # untar_data(URI)

    # 2. Read data and save with 'normal' format: text, label
    # texts, labels, label_index = parse_text_data()
    # df = pd.DataFrame.from_dict({'text': texts, 'label': labels})
    # df.to_csv('./data/20_newsgroup.csv', index=None)

    # 3. Tokenize text to create vocabulary
    df = pd.read_csv('./data/20_newsgroup.csv')

    tokens = tokenizer(df[:10]['text'].tolist())
    vocab = Vocab.create(tokens, max_vocab=1000, min_freq=2)
    print(vocab.itos)
    print(vocab.stoi)

    # 4. create embedding matrix from pretrained word vectors



