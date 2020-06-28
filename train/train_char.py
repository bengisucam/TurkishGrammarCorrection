import logging
import os
import time

import torch
import torchtext
import yaml
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import FastText

import sys


sys.path.append("/content/drive/My Drive/TurkishGrammarCorrection/")

from train.predict import predict

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



logging.basicConfig(level=logging.INFO)

config = parse_yaml('Configuration/config.yaml')

logging.info(config)
max_length = config['dataset']['max_length']

src = SourceField()
tgt = TargetField()
chars = SourceField(tokenize=tokenize)
tv_datafields = [('id', None), ("src", src),
                 ('tgt', tgt)]  # we won't be needing the id, so we pass in None as the field

train, dev, test = torchtext.data.TabularDataset.splits(
    path=config['dataset']['path'], train=config['dataset']['train'],
    validation=config['dataset']['dev'], test=config['dataset']['test'], format='csv', skip_header=True,
    fields=tv_datafields)
___, __, _ = torchtext.data.TabularDataset.splits(
    path=config['dataset']['path'], train=config['dataset']['train'],
    validation=config['dataset']['dev'], test=config['dataset']['test'], format='csv', skip_header=True,
    fields=[('id', None), ("src", chars), ('tgt', None)])

embeddings = None
if bool(config['dataset']['word_embeddings']['use']):
    embeddings = FastText(language='tr', cache='vectors')
max_vocab_size = int(config['dataset']['max_vocab'])

tgt.build_vocab(train, max_size=max_vocab_size, vectors=embeddings)
src.build_vocab(train, max_size=max_vocab_size, vectors=embeddings)
src.vocab.extend(tgt.vocab)
tgt.vocab.extend(src.vocab)

chars.build_vocab(___, max_size=max_vocab_size)
logging.info(f'Train size: {len(train)}\nTest size: {len(test)}\nEval size: {len(dev)}')
logging.info(f'- src vocab size: {len(src.vocab)}')
logging.info(f'- tgt vocab size: {len(tgt.vocab)}')
logging.info(f'- chars vocab size: {len(chars.vocab)}')
input_vocab = src.vocab
output_vocab = tgt.vocab

logging.info(chars.vocab.stoi)

device = config['model']['device']
if not torch.cuda.is_available():
    device = 'cpu'

weight = torch.ones(len(tgt.vocab)).to(torch.device(device))
pad = tgt.vocab.stoi[tgt.pad_token]

loss = NLLLoss(weight, pad)
if device == 'cuda' and torch.cuda.is_available():
    loss.cuda()
logging.info('- creating encoder-decoder')
bidirectional = bool(config['model']['bidirectional'])

hidden_size = config['model']['encoder_lstm_out_decoder_in']
bilstm_hidden_size = int(hidden_size / 2)

bilstm = BiLSTM(bilstm_hidden_size, config['model']['char_embedding_size'], int(config['model']['n_layers']), chars,
                src)
encoder = EncoderRNN(len(src.vocab),
                     max_len=max_length,
                     embedding_total_size=int(config['model']['encoder_lstm_out_decoder_in']) + int(
                         config['model']['word_embedding_size']),
                     hidden_size=hidden_size,
                     n_layers=int(config['model']['n_layers']),
                     rnn_cell=config['model']['rnn_cell'],
                     bidirectional=bidirectional,
                     dropout_p=float(config['model']['dropout_output']),
                     input_dropout_p=float(config['model']['dropout_input']),
                     variable_lengths=config['model']['variable_lengths'],
                     weights=src.vocab.vectors,
                     update_embedding=bool(config['dataset']['word_embeddings']['update']))
decoder = DecoderRNN(len(tgt.vocab), max_length, hidden_size * 2 if bidirectional else hidden_size,
                     embedding_total_size=int(config['model']['encoder_lstm_out_decoder_in']) + int(
                         config['model']['word_embedding_size']),
                     n_layers=int(config['model']['n_layers']),
                     rnn_cell=str(config['model']['rnn_cell']),
                     dropout_p=float(config['model']['dropout_output']),
                     input_dropout_p=float(config['model']['dropout_input']),
                     use_attention=bool(config['model']['use_attention']),
                     bidirectional=bidirectional,
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id,
                     embedder=bilstm,
                     weights=tgt.vocab.vectors,
                     update_embedding=bool(config['dataset']['word_embeddings']['update']))

seq2seq = Seq2seq(encoder, decoder)

if device == 'cuda':
    seq2seq.cuda()
for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)
for param in bilstm.parameters():
    param.data.uniform_(-0.08, 0.08)

lr = config['train']['lr']
optimizer = Optimizer(torch.optim.Adam(list(seq2seq.parameters()) + list(bilstm.parameters()), lr=lr), max_grad_norm=5)
if config['model']['scheduler']['enabled']:
    scheduler = StepLR(optimizer.optimizer, config['model']['scheduler']['rate'])
    optimizer.set_scheduler(scheduler)
logging.info('- starting training')
logging.info(f'- device {device}\n')

t = SupervisedTrainer(loss=loss,
                      batch_size=int(config['train']['batch_size']),
                      print_every=int(config['train']['print_every']),
                      early_stop_threshold=int(config['train']['early_stop_threshold']),
                      save_name="save_name",
                      checkpoint_every=int(config['train']['checkpoint_every']))
logging.info(
    f'Allocated GPU Mem: {torch.cuda.memory_allocated(0)} - Cached Mem: {torch.cuda.memory_cached(0)} - Free Mem: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)}')

seq2seq, path = t.train(seq2seq, bilstm, train, dev_data=dev, test_data=test,
                        num_epochs=config['train']['epoch'],
                        optimizer=optimizer,
                        teacher_forcing_ratio=config['train']['teacher_forcing_ratio'], deviceName=device)
logging.shutdown()
if device == 'cuda':
    torch.cuda.empty_cache()
logging.info("- emptying cuda cache")
time.sleep(5)

test_path = os.path.join(config['dataset']['path'], config['dataset']['test'])
predict(seq2seq, bilstm, input_vocab, output_vocab, test_path, path,
        max_len=max_length, n=500, device='cuda')
