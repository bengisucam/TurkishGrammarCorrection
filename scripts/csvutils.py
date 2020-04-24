import argparse

import pandas as pd


def merge2CSV(inputs, labels, out_path):
    if len(inputs) is not len(labels):
        print("Label and Input size does not match!")
        exit(1)
    df = pd.DataFrame(columns=labels)
    for i in range(len(inputs)):
        file = open(inputs[i], encoding='utf-8', mode='r')
        lines = file.read().strip().split('\n')
        df[labels[i]] = lines
        file.close()
    df.to_csv(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', '--list', dest='files', nargs='+', help='List of files to merge', required=True)
    parser.add_argument('--l', '--list', dest='labels', nargs='+', help='Labels of files to merge', required=True)
    parser.add_argument('--out', action='store', dest='out', help='Out path with .txt at the end')
    opt = parser.parse_args()

    merge2CSV(opt.files, opt.labels, opt.out)
