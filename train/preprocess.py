# remove all quoting marks!
# lower all source beginning, upper all target beginning
# check if sentences start with whitespace
# sourcedaki tüm noktalamaları kaldır.

# manual fix
#  ¼, ˆ
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
    return src


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

        prepsrc = preprocess(src)
        preptgt = preprocess(tgt)

        if prepsrc is None or preptgt is None:
            continue
        srcs.append(prepsrc)
        tgts.append(preptgt)

    dfn = pd.DataFrame(columns=['src', 'tgt'])
    dfn['src'] = srcs
    dfn['tgt'] = tgts
    dfn.to_csv('deneme.csv', index_label='id')
