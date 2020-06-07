import numpy as np

char2ind = {'<unk>': 0, '<pad>': 1, 'a': 2, 'e': 3, 'i': 4, 'n': 5, 'r': 6,
            'l': 7, 'd': 8, 'k': 9, 'ı': 10, 't': 11, 'y': 12, 'm': 13,
            's': 14, 'u': 15, 'o': 16, 'b': 17, '.': 18, 'ü': 19, 'ş': 20,
            'z': 21, 'g': 22, 'ç': 23, 'c': 24, 'h': 25, 'v': 26, 'ğ': 27,
            'p': 28, 'ö': 29, ',': 30, 'f': 31, '’': 32, '0': 33, '̇': 34,
            '1': 35, '2': 36, '5': 37, '3': 38, '9': 39, '?': 40, 'j': 41,
            '4': 42, '7': 43, '6': 44, '8': 45, '!': 46, 'w': 47, ')': 48,
            '(': 49, ';': 50, 'â': 51, 'x': 52, '/': 53, '%': 54, 'q': 55,
            '&': 56, 'î': 57, '=': 58, '$': 59, '#': 60, '>': 61, '@': 62,
            'û': 63, 'ô': 64, '<': 65, '°': 66, 'é': 67, '‘': 68}

ind2char = {i: ch for ch, i in char2ind.items()}

sentence_src = 'belkide çöpler o kadar pis değildir .'
sentence_tgt = 'belki de çöpler o kadar pis değildir .'


def encode_words(sentence, lookup):
    sentence_tokens = []
    tokens = sentence.split()
    print(tokens)
    for t in tokens:
        encoded = np.array([lookup[ch] for ch in t])
        sentence_tokens.append(encoded)
    return np.array(sentence_tokens)


def create_onehot(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


''' TRAIN '''
n_seqs = 1  # num of batch (token steps) ( 1 kelime için)
n_steps = 1  # num of char steps ( her char için )
total_chars = len(char2ind)


def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    print(arr)

    for n in range(0, arr.shape[1], n_steps):
        print(" %d . karakter embedi" % n)

        # The features
        x = arr[:, n:n + n_steps]

        # The targets, shifted by one
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


sentence_arr = encode_words(sentence_src, char2ind)
print(sentence_arr)


for x, y in get_batches(sentence_arr[0], n_seqs, n_steps):
    onehot_embed = create_onehot(arr=x, n_labels=total_chars)
    print(onehot_embed)
