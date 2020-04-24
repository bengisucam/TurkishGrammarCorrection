import argparse


def preprocess(doc: str):
    doc = doc.replace('’', '\'')
    doc = doc.replace('*', '')
    doc = doc.replace('(', '')
    doc = doc.replace(')', '')
    doc = doc.replace(' yada ',' ya da ')
    doc = doc.replace(' YADA ',' YA DA ')
    doc = doc.replace(' Yada ',' Ya da ')
    doc = doc.replace('"','')

    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', action='store', dest='inp',
                        help='Path to train data')
    opt = parser.parse_args()

    f = open(opt.inp, encoding='utf-8', mode='r')
    out = preprocess(f.read().strip())
    f.close()
    with open(opt.inp, mode='w', encoding="utf-8") as outfile:
        outfile.write(out)
        outfile.close()
