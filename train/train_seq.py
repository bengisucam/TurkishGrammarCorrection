import logging
import sys
import time

import torch
import torchtext
import yaml
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import FastText


sys.path.append("/content/drive/My Drive/TurkishGrammarCorrection/")

from seq2seq.dataset import TargetField, SourceField
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.optim import Optimizer
from seq2seq.trainer import SupervisedTrainer
from train.predict import predict


def parse_yaml(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def train(config_file, save_dir, save_name):
    config = parse_yaml(config_file)

    max_length = config['dataset']['max_length']
    hidden_size = config['model']['hidden_size']

    src = SourceField()
    tgt = TargetField()
    tv_datafields = [('id', None), ("src", src),
                     ('tgt', tgt)]  # we won't be needing the id, so we pass in None as the field

    train, dev, test = torchtext.data.TabularDataset.splits(
        path=config['dataset']['path'], train=config['dataset']['train'],
        validation=config['dataset']['dev'], test=config['dataset']['test'], format='csv', skip_header=True,
        fields=tv_datafields)
    embeddings = None
    if bool(config['dataset']['embeddings']['use']):
        embeddings = FastText(language='tr')
    max_vocab_size = int(config['dataset']['max_vocab'])
    src.build_vocab(train, max_size=max_vocab_size, vectors=embeddings)
    tgt.build_vocab(train, max_size=max_vocab_size, vectors=embeddings)

    logging.info('- src vocab size: {}'.format(len(src.vocab)))
    logging.info('- tgt vocab size: {}'.format(len(tgt.vocab)))
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
                         dropout_p=float(config['model']['dropout_output']),
                         input_dropout_p=float(config['model']['dropout_input']),
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
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=lr), max_grad_norm=5)
    if config['model']['scheduler']['enabled']:
        scheduler = StepLR(optimizer.optimizer, config['model']['scheduler']['rate'])
        optimizer.set_scheduler(scheduler)
    logging.info('- starting training')
    logging.info(f'- device {device}\n')
    t = SupervisedTrainer(loss=loss, batch_size=int(config['train']['batch_size']),
                          print_every=int(config['train']['print_every']),
                          early_stop_threshold=int(config['train']['early_stop_threshold']),
                          save_name=save_name,
                          checkpoint_every=int(config['train']['checkpoint_every']))
    seq2seq = t.train(seq2seq, train, dev_data=dev,test_data=test,
                      num_epochs=config['train']['epoch'],
                      optimizer=optimizer,
                      teacher_forcing_ratio=config['train']['teacher_forcing_ratio'], deviceName=device)

    logging.shutdown()
    if device == 'cuda':
        torch.cuda.empty_cache()
    logging.info("- emptying cuda cache")
    time.sleep(5)
    # predict(seq2seq, input_vocab, output_vocab, config['dataset']['test_path'], save_dir + "/" + save_name,
    #         max_len=max_length, n=500)
    logging.info("- saved predictions to {}".format(save_dir + "/" + save_name))

