import logging

import numpy
import torch
import torchtext
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import FastText

from deneme import BiLSTM
from seq2seq.dataset import SourceField, TargetField
from seq2seq.loss import NLLLoss
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.optim import Optimizer
from seq2seq.trainer import SupervisedTrainer
from train.train_seq import parse_yaml


def tokenize(sentence):
    return list(" ".join(sentence.split()))


logging.basicConfig(level=logging.INFO)

config = parse_yaml('Configuration/config.yaml')

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

max_vocab_size = int(config['dataset']['max_vocab'])
src.build_vocab(train, max_size=max_vocab_size)
tgt.build_vocab(train, max_size=max_vocab_size)
chars.build_vocab(___, max_size=max_vocab_size)
print('- src vocab size: {}'.format(len(src.vocab)))
print('- tgt vocab size: {}'.format(len(tgt.vocab)))
print('- chars vocab size: {}'.format(len(chars.vocab)))
input_vocab = src.vocab
output_vocab = tgt.vocab

print(chars.vocab.stoi)

device = config['model']['device']
if not torch.cuda.is_available():
    device = 'cpu'

weight = torch.ones(len(tgt.vocab)).to(torch.device(device))
pad = tgt.vocab.stoi[tgt.pad_token]

loss = NLLLoss(weight, pad)
if device == 'cuda' and torch.cuda.is_available():
    loss.cuda()
print('- creating encoder-decoder')

hidden_size = config['model']['hidden_size']
bilstm_hidden_size = hidden_size
bilstm_embed_size = 128
bilstm_layers = 2

bilstm = BiLSTM(bilstm_hidden_size, bilstm_embed_size, bilstm_layers, chars, src)
bidirectional = bool(config['model']['bidirectional'])
encoder = EncoderRNN(len(src.vocab), max_length, hidden_size * 2,
                     n_layers=int(config['model']['n_layers']),
                     rnn_cell=config['model']['rnn_cell'],
                     bidirectional=bidirectional,
                     dropout_p=float(config['model']['dropout_output']),
                     input_dropout_p=float(config['model']['dropout_input']),
                     variable_lengths=config['model']['variable_lengths'],
                     )
decoder = DecoderRNN(len(tgt.vocab), max_length, hidden_size * 2,
                     n_layers=int(config['model']['n_layers']),
                     rnn_cell=str(config['model']['rnn_cell']),
                     dropout_p=float(config['model']['dropout_output']),
                     input_dropout_p=float(config['model']['dropout_input']),
                     use_attention=bool(config['model']['use_attention']),
                     bidirectional=bidirectional,
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id,
                     embedder=bilstm)

seq2seq = Seq2seq(bilstm, encoder, decoder)


if device == 'cuda':
    seq2seq.cuda()
for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)

lr = config['train']['lr']
optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=lr), max_grad_norm=5)
if config['model']['scheduler']['enabled']:
    scheduler = StepLR(optimizer.optimizer, config['model']['scheduler']['rate'])
    optimizer.set_scheduler(scheduler)
print('- starting training')
print(f'- device {device}\n')

t = SupervisedTrainer(loss=loss, batch_size=int(config['train']['batch_size']),
                      print_every=int(config['train']['print_every']),
                      early_stop_threshold=int(config['train']['early_stop_threshold']),
                      save_name="save_name",
                      checkpoint_every=int(config['train']['checkpoint_every']))
print(f'Allocated GPU Mem: {torch.cuda.memory_allocated(0)} - Cached Mem: { torch.cuda.memory_cached(0)} - Free Mem: {torch.cuda.get_device_properties(0).total_memory -torch.cuda.memory_allocated(0)}')

seq2seq = t.train(seq2seq, train, dev_data=dev, test_data=test,
                  num_epochs=config['train']['epoch'],
                  optimizer=optimizer,
                  teacher_forcing_ratio=config['train']['teacher_forcing_ratio'], deviceName=device)

logging.shutdown()
if device == 'cuda':
    torch.cuda.empty_cache()
print("- emptying cuda cache")
# time.sleep(5)
# predict(seq2seq, input_vocab, output_vocab, config['dataset']['test_path'], save_dir + "/" + save_name,
#         max_len=max_length, n=500)
# logging.info("- saved predictions to {}".format(save_dir + "/" + save_name))
