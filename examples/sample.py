#!/usr/bin/env python
import sys
from builtins import input

import os
import time
import logging
sys.path.append("/content/drive/My Drive/GrammarCorr/TurkishGrammarCorrection/")

import torch
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import FastText

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.data import Seq2SeqDataset
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

LOG_FORMAT = '%(asctime)s:%(name)s:%(levelname)s: %(message)s'



def sample(
        train_source,
        train_target,
        dev_source,
        dev_target,
        experiment_directory,
        checkpoint,
        resume,
        log_level,
):
    """
    # Sample usage

        TRAIN_SRC=data/toy_reverse/train/src.txt
        TRAIN_TGT=data/toy_reverse/train/tgt.txt
        DEV_SRC=data/toy_reverse/dev/src.txt
        DEV_TGT=data/toy_reverse/dev/tgt.txt

    ## Training
    ```shell
    $ ./examples/sample.py $TRAIN_SRC $TRAIN_TGT $DEV_SRC $DEV_TGT -expt
    $EXPT_PATH
    ```
    ## Resuming from the latest checkpoint of the experiment
    ```shell
    $ ./examples/sample.py $TRAIN_SRC $TRAIN_TGT $DEV_SRC $DEV_TGT -expt
    $EXPT_PATH -r
    ```
    ## Resuming from a specific checkpoint
    ```shell
    $ python examples/sample.py $TRAIN_SRC $TRAIN_TGT $DEV_SRC $DEV_TGT -expt
    $EXPT_PATH -c $CHECKPOINT_DIR
    ```
    """
    logging.basicConfig(
        format=LOG_FORMAT,
        level=getattr(logging, log_level.upper()),
    )
    logging.info('train_source: %s', train_source)
    logging.info('train_target: %s', train_target)
    logging.info('dev_source: %s', dev_source)
    logging.info('dev_target: %s', dev_target)
    logging.info('experiment_directory: %s', experiment_directory)

    if checkpoint:
        seq2seq, input_vocab, output_vocab = load_checkpoint(
            experiment_directory, checkpoint)
    else:
        seq2seq, input_vocab, output_vocab = train_model(
            train_source,
            train_target,
            dev_source,
            dev_target,
            experiment_directory,
            resume=resume,
        )

    predictor = Predictor(seq2seq, input_vocab, output_vocab)

    while True:
        seq_str = input('Type in a source sequence: ')
        seq = seq_str.strip().split()
        print(predictor.predict(seq))


def load_checkpoint(experiment_directory, checkpoint):
    checkpoint_path = os.path.join(
        experiment_directory,
        Checkpoint.CHECKPOINT_DIR_NAME,
        checkpoint,
    )
    logging.info('Loading checkpoint from {}'.format(
        checkpoint_path,
    ))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    return seq2seq, input_vocab, output_vocab


def train_model(
        train_source,
        train_target,
        dev_source,
        dev_target,
        experiment_directory,
        resume=False,
):
    # Prepare dataset
    train = Seq2SeqDataset.from_file(train_source, train_target)
    embeddings = FastText(language='tr', cache='../train/vectors')
    train.build_vocab(500000, 500000,embeddings)
    dev = Seq2SeqDataset.from_file(
        dev_source,
        dev_target,
        share_fields_from=train,
    )
    input_vocab = train.src_field.vocab
    output_vocab = train.tgt_field.vocab

    # Prepare loss
    weight = torch.ones(len(output_vocab))
    pad = output_vocab.stoi[train.tgt_field.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not resume:
        seq2seq, optimizer, scheduler = initialize_model(
            train, input_vocab, output_vocab,input_vocab.vectors)

    # Train
    trainer = SupervisedTrainer(
        loss=loss,
        batch_size=32,
        checkpoint_every=1000,
        print_every=10,
        experiment_directory=experiment_directory,
    )
    start = time.clock()
    try:
        seq2seq = trainer.train(
            seq2seq,
            train,
            n_epochs=20,
            dev_data=dev,
            optimizer=optimizer,
            teacher_forcing_ratio=1.0,
            resume=resume,
        )
    # Capture ^C
    except KeyboardInterrupt:
        pass
    end = time.clock() - start
    logging.info('Training time: %.2fs', end)

    return seq2seq, input_vocab, output_vocab


def initialize_model(
        train,
        input_vocab,
        output_vocab,
        embeddings,
        max_len=16,
        hidden_size=300,
        dropout_p=0.1,
        bidirectional=True,
):
    # Initialize model
    encoder = EncoderRNN(
        len(input_vocab),
        max_len,
        hidden_size,
        n_layers=2,
        embedding=embeddings,
        bidirectional=bidirectional,
        variable_lengths=True,
    )
    decoder = DecoderRNN(
        len(output_vocab),
        max_len,
        hidden_size * (2 if bidirectional else 1),
        n_layers=2,
        dropout_p=dropout_p,
        use_attention=True,
        bidirectional=bidirectional,
        eos_id=train.tgt_field.eos_id,
        sos_id=train.tgt_field.sos_id,
    )
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq = seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    # Optimizer and learning rate scheduler can be customized by
    # explicitly constructing the objects and pass to the trainer
    optimizer = Optimizer(
        torch.optim.Adam(seq2seq.parameters(),lr=0.0012), max_grad_norm=5)
    scheduler = StepLR(optimizer.optimizer, 1)
    optimizer.set_scheduler(scheduler)

    return seq2seq, optimizer, scheduler


if __name__ == '__main__':
    TRAIN_SRC="../data/source.txt"
    TRAIN_TGT="../data/target.txt"
    DEV_SRC="../data/traintest/source.txt"
    DEV_TGT="../data/traintest/target.txt"
    sample(TRAIN_SRC,TRAIN_TGT,DEV_SRC,DEV_TGT,"../expt",None,False,"info")
