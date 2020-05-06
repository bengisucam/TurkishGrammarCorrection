import logging

import fasttext
import fasttext.util as util
import numpy
import torch


class Model(object):
    def __init__(self, path='../cc.tr.300.bin'):
        self.ft = fasttext.load_model(path)

    def get_sentence_vector(self, sentence):
        return self.ft.get_sentence_vector(sentence)

    def get_word_vector(self, word):
        return self.ft.get_word_vector(word)

    def toVecList(self, input_vocab):
        logging.info('- creating embeddings')
        device=torch.device('cuda')
        lst = []
        for tok in input_vocab.itos:
            lst.append(self.get_word_vector(tok))
        return torch.from_numpy(numpy.array(lst)).float().to(device)

    def reduce_dim(self, size):
        logging.info("\n- reducing fast text model dimension to {}".format(size))
        self.ft = util.reduce_model(self.ft, size)
        return self
