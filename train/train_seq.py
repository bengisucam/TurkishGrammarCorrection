import argparse
import glob
import logging
import os
import time

import fasttext
import numpy
import torch
import torchtext
import yaml
from fasttext import util
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import FastText

from seq2seq.dataset import TargetField, SourceField
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.optim import Optimizer
from seq2seq.trainer import SupervisedTrainer
from seq2seq.util.checkpoint import Checkpoint
from train.predict import predict


class Model(object):
    def __init__(self, path='../cc.tr.300.bin'):
        self.ft = fasttext.load_model(path)

    def get_sentence_vector(self, sentence):
        return self.ft.get_sentence_vector(sentence)

    def get_word_vector(self, word):
        return self.ft.get_word_vector(word)

    def toVecList(self, input_vocab):
        logging.info('- creating embeddings')
        device = torch.device('cuda')
        lst = []
        for tok in input_vocab.itos:
            lst.append(self.get_word_vector(tok))
        return torch.from_numpy(numpy.array(lst)).float().to(device)

    def reduce_dim(self, size):
        logging.info("\n- reducing fast text model dimension to {}".format(size))
        self.ft = util.reduce_model(self.ft, size)
        return self


def parse_yaml(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_save_name(config_):
    return str(config_['model']['rnn_cell']) + "_" + str(config_['model']['hidden_size']) + "_" + str(
        config_['model']['loss']) + "_" + str(config_['model']['optimizer']) + "_" + str(
        config_['train']['lr']) + "_" + str(
        config_['train']['epoch']) + \
           "_bidir - " + str(config_['model']['bidirectional']) + "_" + "attention - " + str(
        config_['model']['bidirectional']) + "_update_embd - " + str(
        config_['dataset']['embeddings']['update']) + "_variable - " + str(
        config_['model']['variable_lengths']) + "_nlayers - " + str(
        config_['model']['n_layers']) + "_schRate - " + str(config_['model']['scheduler']['rate'])


def train(config_file, save_dir):
    config = parse_yaml(config_file)

    max_length = config['dataset']['max_length']
    hidden_size = config['model']['hidden_size']

    save_name = get_save_name(config)
    if not os.path.exists(save_dir + "/" + save_name):
        os.makedirs(save_dir + "/" + save_name)

    src = SourceField()
    tgt = TargetField()
    tv_datafields = [('id', None), ("src", src),
                     ('tgt', tgt)]  # we won't be needing the id, so we pass in None as the field
    train = torchtext.data.TabularDataset(
        path=config['dataset']['train_path'], format='csv',
        fields=tv_datafields,
        skip_header=True
    )
    dev = torchtext.data.TabularDataset(
        path=config['dataset']['dev_path'], format='csv',
        fields=tv_datafields,
        skip_header=True
    )
    embeddings = None
    if bool(config['dataset']['embeddings']['use']):
        embeddings = FastText(language='tr')
        embeddings.dim = hidden_size
    max_vocab_size = int(config['dataset']['max_vocab'])
    src.build_vocab(train, max_size=max_vocab_size, vectors=embeddings)
    tgt.build_vocab(train, max_size=max_vocab_size)

    logging.info('- vocab size: {}'.format(len(src.vocab)))
    logging.info('- embedding size: {}'.format(src.vocab.vectors.size()))
    input_vocab = src.vocab
    output_vocab = tgt.vocab
    #

    device = config['model']['device']
    if not torch.cuda.is_available():
        device = 'cpu'

    weight = torch.ones(len(tgt.vocab)).to(torch.device(device))
    pad = tgt.vocab.stoi[tgt.pad_token]

    if str(config['model']['loss']) == 'Perp':
        loss = Perplexity(weight, pad)
    else:
        loss = NLLLoss(weight, pad)
    if device == 'cuda' and torch.cuda.is_available():
        loss.cuda()
    logging.info('- creating encoder-decoder')

    bidirectional = bool(config['model']['bidirectional'])
    encoder = EncoderRNN(len(src.vocab), max_length, hidden_size,
                         n_layers=int(config['model']['n_layers']),
                         rnn_cell=config['model']['rnn_cell'],
                         bidirectional=bidirectional,
                         dropout_p=float(config['model']['dropout_output']),
                         input_dropout_p=float(config['model']['dropout_input']),
                         variable_lengths=config['model']['variable_lengths'],
                         embedding=src.vocab.vectors,
                         update_embedding=bool(config['dataset']['embeddings']['update']))
    decoder = DecoderRNN(len(tgt.vocab), max_length, hidden_size * 2 if bidirectional else hidden_size,
                         n_layers=int(config['model']['n_layers']),
                         rnn_cell=str(config['model']['rnn_cell']),
                         # dropout_p=float(config['model']['dropout_output']),
                         # input_dropout_p=float(config['model']['dropout_input']),
                         use_attention=bool(config['model']['use_attention']),
                         bidirectional=bidirectional,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    if device == 'cuda':
        seq2seq.cuda()
    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    lr = config['train']['lr']
    if str(config['model']['optimizer']) == 'SGD':
        optimizer = Optimizer(torch.optim.SGD(seq2seq.parameters(), lr=lr), max_grad_norm=5)
    else:
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=lr))
    if config['model']['scheduler']['enabled']:
        scheduler = StepLR(optimizer.optimizer, config['model']['scheduler']['rate'])
        optimizer.set_scheduler(scheduler)
    logging.info('- starting training\n')
    t = SupervisedTrainer(loss=loss, batch_size=int(config['train']['batch_size']),
                          print_every=int(config['train']['print_every']),
                          early_stop_threshold=int(config['train']['early_stop_threshold']))
    seq2seq = t.train(seq2seq, train, dev_data=dev,
                      num_epochs=config['train']['epoch'],
                      optimizer=optimizer,
                      teacher_forcing_ratio=config['train']['teacher_forcing_ratio'], deviceName=device)

    saveDir = Checkpoint(model=seq2seq,
                         optimizer=optimizer,
                         epoch=0, step=0,
                         input_vocab=input_vocab,
                         output_vocab=output_vocab).save(save_dir, save_name)
    logging.info('- saved models to {}'.format(save_dir))
    logging.shutdown()
    if device == 'cuda':
        torch.cuda.empty_cache()
    logging.info("- emptying cuda cache")
    time.sleep(5)
    predict(seq2seq, input_vocab, output_vocab, config['dataset']['test_path'], save_dir + "/" + save_name,
            max_len=max_length, n=500)
    logging.info("- saved predictions to {}".format(save_dir + "/" + save_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', dest='config',
                        help='Path to train yaml config file')
    parser.add_argument('--save', action='store', dest='save',
                        help='Save dir')
    opt = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config_path = opt.config

    for config_file in glob.glob(config_path + "/*.yaml"):
        logging.info("- using {}".format(config_file))
        train(config_file, opt.save)
