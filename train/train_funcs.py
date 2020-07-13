# -*- coding: utf-8 -*-
import logging
import os
import time

import dill
import torch
import torchtext
import yaml
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import FastText

import sys

from seq2seq.evaluator import Predictor, Evaluator
from zemberek_python.base import ZemberekPython

sys.path.append("/content/drive/My Drive/TurkishGrammarCorrection/")
sys.path.append('C:/Users/furka/Desktop/TurkishGrammarCorrection')
sys.path.append('C:/Users/furka/Desktop/TurkishGrammarCorrection/seq2seq')

from seq2seq.util.checkpoint import Checkpoint

from seq2seq.models.bilstm import BiLSTM
from seq2seq.dataset import SourceField, TargetField
from seq2seq.loss import NLLLoss
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.optim import Optimizer
from seq2seq.trainer import SupervisedTrainer


def parse_yaml(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def tokenize(sentence):
    return list(" ".join(sentence.split()))


def load_vocabs(configuration):
    path = configuration['model']['pre_trained_model']
    return Checkpoint.load_vocabs(path)


def load_models(configuration, input_voc, output_voc, chars_vocab):
    max_length = configuration['dataset']['max_length']
    device = configuration['model']['device']
    if not torch.cuda.is_available():
        device = 'cpu'
    path = configuration['model']['pre_trained_model']
    bidirectional = bool(configuration['model']['bidirectional'])

    hidden_size = configuration['model']['encoder_lstm_out_decoder_in']
    bilstm_hidden_size = int(hidden_size / 2)

    bilstm = BiLSTM(bilstm_hidden_size, configuration['model']['char_embedding_size'],
                    int(configuration['model']['n_layers']), chars_vocab,
                    input_voc, device=torch.device(device))  # src

    bilstm_state_dict, seq2seq_state_dict = Checkpoint.load_model_states(path, device)
    encoder_word_vectors = seq2seq_state_dict['encoder.word_embedding.weight']
    decoder_word_vectors = seq2seq_state_dict['decoder.word_embedding.weight']
    encoder = EncoderRNN(len(input_voc),
                         max_len=max_length,
                         embedding_total_size=int(configuration['model']['encoder_lstm_out_decoder_in'])
                                              + int(configuration['model']['word_embedding_size']),
                         hidden_size=hidden_size,
                         n_layers=int(configuration['model']['n_layers']),
                         rnn_cell=configuration['model']['rnn_cell'],
                         bidirectional=bidirectional,
                         dropout_p=float(configuration['model']['dropout_output']),
                         input_dropout_p=float(configuration['model']['dropout_input']),
                         variable_lengths=configuration['model']['variable_lengths'],
                         weights=encoder_word_vectors,
                         update_embedding=bool(configuration['dataset']['word_embeddings']['update']))
    decoder = DecoderRNN(len(output_voc), max_length, hidden_size * 2 if bidirectional else hidden_size,
                         embedding_total_size=int(configuration['model']['encoder_lstm_out_decoder_in'])
                                              + int(configuration['model']['word_embedding_size']),
                         n_layers=int(configuration['model']['n_layers']),
                         rnn_cell=str(configuration['model']['rnn_cell']),
                         dropout_p=float(configuration['model']['dropout_output']),
                         input_dropout_p=float(configuration['model']['dropout_input']),
                         use_attention=bool(configuration['model']['use_attention']),
                         bidirectional=bidirectional,
                         eos_id=output_voc.stoi['<eos>'], sos_id=output_voc.stoi['<sos>'],
                         weights=decoder_word_vectors,
                         update_embedding=bool(configuration['dataset']['word_embeddings']['update'])
                         , device=torch.device(device))
    seq2seq = Seq2seq(encoder, decoder)

    seq2seq.load_state_dict(seq2seq_state_dict)
    bilstm.load_state_dict(bilstm_state_dict)
    return bilstm, seq2seq


def initialize_models(configuration, src, tgt, chars):
    logging.info('- creating encoder-decoder')

    input_vocab = src.vocab
    output_vocab = tgt.vocab

    max_length = configuration['dataset']['max_length']

    device = configuration['model']['device']
    if not torch.cuda.is_available():
        device = 'cpu'

    bidirectional = bool(configuration['model']['bidirectional'])
    hidden_size = configuration['model']['encoder_lstm_out_decoder_in']
    bilstm_hidden_size = int(hidden_size / 2)

    bilstm = BiLSTM(bilstm_hidden_size, configuration['model']['char_embedding_size'],
                    int(configuration['model']['n_layers']), chars,
                    src)  # src
    encoder = EncoderRNN(len(input_vocab),
                         max_len=max_length,
                         embedding_total_size=int(configuration['model']['encoder_lstm_out_decoder_in'])
                                              + int(configuration['model']['word_embedding_size']),
                         hidden_size=hidden_size,
                         n_layers=int(configuration['model']['n_layers']),
                         rnn_cell=configuration['model']['rnn_cell'],
                         bidirectional=bidirectional,
                         dropout_p=float(configuration['model']['dropout_output']),
                         input_dropout_p=float(configuration['model']['dropout_input']),
                         variable_lengths=configuration['model']['variable_lengths'],
                         weights=src.vocab.vectors,
                         update_embedding=bool(configuration['dataset']['word_embeddings']['update']))
    decoder = DecoderRNN(len(output_vocab), max_length, hidden_size * 2 if bidirectional else hidden_size,
                         embedding_total_size=int(configuration['model']['encoder_lstm_out_decoder_in'])
                                              + int(configuration['model']['word_embedding_size']),
                         n_layers=int(configuration['model']['n_layers']),
                         rnn_cell=str(configuration['model']['rnn_cell']),
                         dropout_p=float(configuration['model']['dropout_output']),
                         input_dropout_p=float(configuration['model']['dropout_input']),
                         use_attention=bool(configuration['model']['use_attention']),
                         bidirectional=bidirectional,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id,
                         weights=tgt.vocab.vectors,
                         update_embedding=bool(configuration['dataset']['word_embeddings']['update']))
    seq2seq = Seq2seq(encoder, decoder)

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)
    for param in bilstm.parameters():
        param.data.uniform_(-0.08, 0.08)

    if device == 'cuda':
        seq2seq.cuda()
        bilstm.cuda()

    return bilstm, seq2seq


def initialize_data(configuration):
    src = SourceField()
    tgt = TargetField()
    chars = SourceField(tokenize=tokenize)
    tv_datafields = [('id', None), ("src", src),
                     ('tgt', tgt)]  # we won't be needing the id, so we pass in None as the field

    train, dev, test = torchtext.data.TabularDataset.splits(
        path=configuration['dataset']['path'], train=configuration['dataset']['train'],
        validation=configuration['dataset']['dev'], test=configuration['dataset']['test'], format='csv',
        skip_header=True,
        fields=tv_datafields)
    ___, __, _ = torchtext.data.TabularDataset.splits(
        path=configuration['dataset']['path'], train=configuration['dataset']['train'],
        validation=configuration['dataset']['dev'], test=configuration['dataset']['test'], format='csv',
        skip_header=True,
        fields=[('id', None), ("src", chars), ('tgt', None)])

    embeddings = None
    if bool(configuration['dataset']['word_embeddings']['use']):
        embeddings = FastText(language='tr', cache='vectors')
    max_vocab_size = int(configuration['dataset']['max_vocab'])

    tgt.build_vocab(train, dev, test, max_size=max_vocab_size, vectors=embeddings)
    src.build_vocab(train, dev, test, max_size=max_vocab_size, vectors=embeddings)
    tgt.vocab.extend(src.vocab)

    chars.build_vocab(___, max_size=max_vocab_size)
    logging.info(f'Train size: {len(train)}\nTest size: {len(test)}\nEval size: {len(dev)}')
    logging.info(f'- src vocab size: {len(src.vocab)}')
    logging.info(f'- tgt vocab size: {len(tgt.vocab)}')
    logging.info(f'- chars vocab size: {len(chars.vocab)}')

    # logging.info(chars.vocab.stoi)

    return src, tgt, chars, train, dev, test


def train(configuration, seq2seq, bilstm, src, tgt, train_set, dev_set, test_set, char_vocab):
    device = configuration['model']['device']
    if not torch.cuda.is_available():
        device = 'cpu'

    weight = torch.ones(len(tgt.vocab)).to(torch.device(device))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = NLLLoss(weight, pad)
    if device == 'cuda' and torch.cuda.is_available():
        loss.cuda()

    device = configuration['model']['device']
    lr = configuration['train']['lr']
    optimizer = Optimizer(torch.optim.Adam(list(seq2seq.parameters()) + list(bilstm.parameters()), lr=lr),
                          max_grad_norm=5)
    if configuration['model']['scheduler']['enabled']:
        scheduler = StepLR(optimizer.optimizer, configuration['model']['scheduler']['rate'])
        optimizer.set_scheduler(scheduler)
    logging.info('- starting training')

    t = SupervisedTrainer(loss=loss,
                          batch_size=int(configuration['train']['batch_size']),
                          print_every=int(configuration['train']['print_every']),
                          early_stop_threshold=int(configuration['train']['early_stop_threshold']),
                          save_name="Exp",
                          checkpoint_every=int(configuration['train']['checkpoint_every']))
    logging.info(
        f'Allocated GPU Mem: {torch.cuda.memory_allocated(0)} - Cached Mem: {torch.cuda.memory_cached(0)} - Free Mem: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)}')

    return t.train(seq2seq, bilstm, train_set, char_vocab, dev_data=dev_set, test_data=test_set,
                   num_epochs=configuration['train']['epoch'],
                   optimizer=optimizer,
                   teacher_forcing_ratio=configuration['train']['teacher_forcing_ratio'], deviceName=device)


# logging.basicConfig(level=logging.INFO)
# config = parse_yaml('C:/Users/furka/Desktop/TurkishGrammarCorrection/train/Configuration/config.yaml')
# logging.info(config)
# train_only = False
# if train_only:
#     src, tgt, chars, train_set, dev_set, test_set = initialize_data(config)
#     bilstm, seq2seq = initialize_models(config, src, tgt, chars)
#     t = train(config, seq2seq, bilstm, train_set, dev_set, test_set, chars)
# else:
#     src, tgt, chars, train_set, dev_set, test_set = initialize_data(config)
#     input_vocab, output_vocab, char_vocab = load_vocabs(config)
#     bilstm, seq2seq = load_models(config, src.vocab, tgt.vocab, chars.vocab)
#     while True:
#         sentence = input()
#     # predict(seq2seq, bilstm, input_vocab, output_vocab, '../data/train/questions/test.csv', None)
#     # sentence = sys.argv[1]
#     # print(sentence)
#         print(predict_single(sentence.lower(), seq2seq, bilstm, input_vocab, output_vocab, 'cuda'))


def load_for_prediction(config):
    input_vocab, output_vocab, char_vocab = load_vocabs(config)
    bilstm, seq2seq = load_models(config, input_vocab, output_vocab, char_vocab)
    return seq2seq, bilstm, input_vocab, output_vocab


def create_predictor(config_path):
    torch.cuda.empty_cache()
    config = parse_yaml(config_path)

    model, bilstm, input_vocab, output_vocab = load_for_prediction(config)
    predictor = Predictor(model, bilstm, input_vocab, output_vocab, device='cpu')

    zemberek = ZemberekPython(config['zemberek_path'])
    zemberek = zemberek.startJVM().CreateTokenizer().CreateTurkishMorphology().CreateSpellChecker().CreateNormalizer(
        config['zemberek_normalizer_path'])
    print('creation successfull')
    return zemberek, predictor


config = parse_yaml('C:/Users/furka/Desktop/TurkishGrammarCorrection/train/Configuration/config.yaml')

src = SourceField()
tgt = TargetField()
chars = SourceField(tokenize=tokenize)
tv_datafields = [('id', None), ("src", src),
                 ('tgt', tgt)]  # we won't be needing the id, so we pass in None as the field

t, dev, test = torchtext.data.TabularDataset.splits(
    path=config['dataset']['path'], train=config['dataset']['train'],
    validation=config['dataset']['dev'], test=config['dataset']['test'], format='csv',
    skip_header=True,
    fields=tv_datafields)
torch.cuda.empty_cache()
model, bilstm, input_vocab, output_vocab = load_for_prediction(config)
predictor = Predictor(model, bilstm, input_vocab, output_vocab, device='cuda')

import pandas as pd

df = pd.read_csv(os.path.join(config['dataset']['path'], config['dataset']['test']), index_col='id')
a = 0
len_output_total = 0
len_tgt_total = 0
correct_total = 0


def correctnum(pred, tgt):
    t = 0
    for w in pred:
        if w in tgt:
            t += 1
    return t

p=0.0

for row in df.iterrows():
    x = row[1]
    src = x.values[0]
    tgt = x.values[1].split()

    pred = predictor.predict(src.split() )
    if pred[-1] == '<eos>':
        pred = pred[:-1]

    len_output_total += len(pred)
    len_tgt_total += len(tgt)


    correct_total += correctnum(pred, tgt)

    if a % 1000 == 0 and a > 0:
        print(a)
    a += 1

precision = float(correct_total) / len_output_total
recall = float(correct_total) / len_tgt_total
f1 = 2*(recall * precision) / (recall + precision)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
