# remove all quoting marks!
# lower all source beginning, upper all target beginning
# check if sentences start with whitespace
# sourcedaki tüm noktalamaları kaldır.

# manual fix
#  ¼, ˆ
import re
import string

replacements = {
    '-': '',
    '\"': '',
    'Ó': '',
    'Œ': '',
    '°': '',
    'ÿ': '',
    '~': '',
    '“': '',
    '´': '\'',
    'º': '',
    'š': 'ü',
    'É': 'e',
    'è': 'è',
    '`': '',
    '¤': 'ğ',
    '\x95': '',
    'Ô': '',
    '¾': 'ş',
    '½': 'ş',
    'ƒ': 'a',
    '‹': 'i',
    '›': 'ı',
    '\x7f': '',
    '» ': '',
    '• ': '',
    '•': '',
    '© ': '',
    'Å': 'a',
    '\x01': '',
    '\x1a': '',
    '\x1b': '',
    ' ‘ ': '\'',
    '‘': '\'',
    '” ': '',
    '”': '',
    '„': '',
    '· ': '',
    '·': '',
    'ù': '',
    '¢': '',
    '™ ': ''

}


def IsInteger(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def check(sentence):
    if check_too_many_hours(sentence) is not None:
        if check_integer_count(sentence) is not None:
            if check_punc(sentence) is not None:
                return sentence
    return None


def check_punc(sentence):
    c = 0
    split_text = sentence.split(' ')
    try:
        if sentence[0] in string.punctuation: return None
    except:
        return None
    x = re.search("^\.\.\..*$", split_text[0])
    if x is not None:
        return None
    if split_text[-1] is ':': return None
    for tok in sentence.split(' '):
        if tok in string.punctuation:
            c += 1
    if c > 2:
        return None
    return sentence


def check_integer_count(sentence):
    c = 0
    first = True
    for tok in sentence.split(' '):
        if IsInteger(tok):
            if first:
                return None
            first = False
            c += 1
    if c > 1:
        return None
    return sentence


def check_too_many_hours(sentence):
    c = 0
    for tok in sentence.split(' '):
        x = re.search("[0-9][0-9][.-][0-9][0-9]", tok)
        if x is not None:
            c += 1
    if c > 1:
        return None
    return sentence


def preprocess(src):
    try:
        for replacement in replacements:
            src = src.replace(replacement, replacements[replacement])
    except AttributeError:
        return None
    max_tok_len = 21
    for tok in src.split(' '):
        if len(tok) >= max_tok_len:
            return None

    return check(src)


# def preprocess_chars(sequence):
#     arr = []
#     for tok in sequence:
#         for replacement in replacements:
#             tok = tok.replace(replacement, replacements[replacement])
#         arr.append(tok)
#     return arr
if __name__ == '__main__':

    import pandas as pd

    srcs = []
    tgts = []
    df = pd.read_csv('../data/newscor/seq2seqdata.csv', index_col='id', encoding='utf-8', warn_bad_lines=True)
    for i in range(len(df)):
        src = df['src'].iloc[i]
        tgt = df['tgt'].iloc[i]

        prepsrc = preprocess(src.lower())
        preptgt = preprocess(tgt.lower())

        if prepsrc is None or preptgt is None:
            continue
        srcs.append(prepsrc)
        tgts.append(preptgt)

    dfn = pd.DataFrame(columns=['src', 'tgt'])
    dfn['src'] = srcs
    dfn['tgt'] = tgts
    dfn.to_csv('deneme.csv', index_label='id')
