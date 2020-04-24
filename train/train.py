import argparse

import torch
import torchtext
from seq2seq.dataset import TargetField, SourceField
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.optim import Optimizer
from seq2seq.trainer import SupervisedTrainer
from seq2seq.util.checkpoint import Checkpoint
from torch.optim.lr_scheduler import StepLR

from scripts.fasttextembeddings import Model
from scripts.yamlutilities import parse_yaml

parser = argparse.ArgumentParser()
parser.add_argument('--conf', action='store', dest='config',
                    help='Path to train yaml config file')
parser.add_argument('--save', action='store', dest='save_dir',
                    help='Path to save model to')
opt = parser.parse_args()

config_path = opt.config
config = parse_yaml(config_path)
max_length = config['dataset']['max_length']
hidden_size = config['model']['hidden_size']


def len_filter(example):
    return max_length >= len(example.src) > 0 and max_length >= len(example.tgt) > 0 and example.tgt != example.src


src = SourceField()
tgt = TargetField()
tv_datafields = [('id', None), ("src", src),
                 ('tgt', tgt)]  # we won't be needing the id, so we pass in None as the field
train = torchtext.data.TabularDataset(
    path=config['dataset']['train_path'], format='csv',
    fields=tv_datafields,
    filter_pred=len_filter, skip_header=True
)

max_vocab_size = int(config['dataset']['max_vocab'])
src.build_vocab(train, max_size=max_vocab_size)
tgt.build_vocab(train, max_size=max_vocab_size)

input_vocab = src.vocab
output_vocab = tgt.vocab

ftModel = Model(config['dataset']['embeddings_path']).reduce_dim(hidden_size)
embeddings = ftModel.toVecList(input_vocab=input_vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.ones(len(tgt.vocab)).to(device)
pad = tgt.vocab.stoi[tgt.pad_token]

if str(config['model']['loss']) == 'Perp':
    loss = Perplexity(weight, pad)
else:
    loss = NLLLoss(weight, pad)
if torch.cuda.is_available():
    loss.cuda()
print('- creating encoder-decoder')
n_layers = int(config['model']['n_layers'])
bidirectional = bool(config['model']['bidirectional'])
encoder = EncoderRNN(len(src.vocab), max_length, hidden_size,
                     n_layers=n_layers,
                     rnn_cell=config['model']['rnn_cell'],
                     bidirectional=bidirectional,
                     variable_lengths=config['model']['variable_lengths'],
                     embedding=embeddings,
                     update_embedding=True)
decoder = DecoderRNN(len(tgt.vocab), max_length, hidden_size * 2 if bidirectional else hidden_size,
                     n_layers=n_layers,
                     rnn_cell=str(config['model']['rnn_cell']),
                     dropout_p=float(config['model']['dropout']),
                     use_attention=config['model']['use_attention'],
                     bidirectional=bidirectional,
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id)
seq2seq = Seq2seq(encoder, decoder)
if torch.cuda.is_available():
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
print('- starting training\n')
t = SupervisedTrainer(loss=loss, batch_size=int(config['train']['batch_size']),
                      print_every=int(config['train']['print_every']))
seq2seq = t.train(seq2seq, train,
                  num_epochs=config['train']['epoch'],
                  optimizer=optimizer,
                  teacher_forcing_ratio=config['train']['teacher_forcing_ratio'])
save_dir = Checkpoint(model=seq2seq,
                      optimizer=optimizer,
                      epoch=0, step=0,
                      input_vocab=input_vocab,
                      output_vocab=output_vocab).save(config['save_dir'], opt.save_dir)
print('- saved models to {}'.format(save_dir))
# predictor = Predictor(seq2seq, input_vocab, output_vocab)
# while True:
#     seq_str = input("Type in a source sequence:")
#     seq = seq_str.strip().split()
#     print(predictor.predict(seq))
