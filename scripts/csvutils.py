import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def merge2CSV(inputs, out_path):
    labels = ['src', 'tgt']
    if len(inputs) is not len(labels):
        print("Label and Input size does not match!")
        exit(1)
    df = pd.DataFrame(columns=labels)
    for i in range(len(inputs)):
        file = open(inputs[i], encoding='utf-8', mode='r')
        lines = file.read().strip().split('\n')
        df[labels[i]] = lines
        file.close()
    df.to_csv(out_path, index_label='id')


def splitCSV(data, n_total):
    df = pd.read_csv(data, index_col='id', quoting=0).sample(n=n_total)
    train, test_dev = train_test_split(df, test_size=0.3)
    test, dev = train_test_split(test_dev, test_size=0.5)
    train.to_csv('../data/train/70-15-15/train.csv')
    test.to_csv('../data/train/70-15-15/test.csv')
    dev.to_csv('../data/train/70-15-15/dev.csv')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--f', '--list', dest='files', nargs='+', help='List of files to merge', required=True)
    # parser.add_argument('--out', action='store', dest='out', help='Out path with .txt at the end')
    # opt = parser.parse_args()
    # merge2CSV(['../data/newscor_processed/source_xab.txt', '../data/newscor_processed/target_xab.txt'],
    #           '../data/train/xab_seq2seq.csv')

    splitCSV('../data/newscor/seq2seqdata.csv',400000)
    exit()
