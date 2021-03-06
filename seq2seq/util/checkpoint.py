from __future__ import print_function

import logging
import os
import time
import shutil
from seq2seq.models.bilstm import BiLSTM
from seq2seq.dataset import SourceField, TargetField
from seq2seq.loss import NLLLoss
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.optim import Optimizer
import torch
import dill


class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.

    Args:
        model (seq2seq): seq2seq model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
        step (int): number of examples seen within the current epoch
        input_vocab (Vocabulary): vocabulary for the input language
        output_vocab (Vocabulary): vocabulary for the output language

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
        INPUT_VOCAB_FILE (str): name of the input vocab file
        OUTPUT_VOCAB_FILE (str): name of the output vocab file
    """

    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'
    BILSTM_NAME = 'bilstm.pt'
    INPUT_VOCAB_FILE = 'input_vocab.pt'
    OUTPUT_VOCAB_FILE = 'output_vocab.pt'
    CHAR_VOCAB_FILE = 'chars_vocab.pt'

    def __init__(self, model, bilstm, optimizer, epoch, step, input_vocab, output_vocab,char_vocab, path=None):
        self.model = model
        self.bilstm = bilstm
        self.optimizer = optimizer
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.char_vocab=char_vocab
        self.epoch = epoch
        self.step = step
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir, save_name):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the Experiments root directory
        Returns:
             str: path to the saved checkpoint subdirectory
             :param experiment_dir:
             :param save_name:
        """
        self._path = os.path.join(experiment_dir, save_name)
        path = self._path

        if not os.path.exists(path):
            # shutil.rmtree(path)
            os.makedirs(path)
        # torch.save({'epoch': self.epoch,
        #             'step': self.step,
        #             'optimizer': self.optimizer
        #             },
        #            os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model.state_dict(), os.path.join(path, self.MODEL_NAME))
        torch.save(self.bilstm.state_dict(), os.path.join(path, self.BILSTM_NAME))

        with open(os.path.join(path, self.INPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.input_vocab, fout)
        with open(os.path.join(path, self.CHAR_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.char_vocab, fout)
        with open(os.path.join(path, self.OUTPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.output_vocab, fout)
        logging.info('- saved models to {}'.format(path))
        return path


    @classmethod
    def load_vocabs(cls,path):
        with open(os.path.join(path, cls.INPUT_VOCAB_FILE), 'rb') as fin:
            input_vocab = dill.load(fin)
        with open(os.path.join(path, cls.OUTPUT_VOCAB_FILE), 'rb') as fin:
            output_vocab = dill.load(fin)
        with open(os.path.join(path, cls.CHAR_VOCAB_FILE), 'rb') as fin:
            chars_vocab = dill.load(fin)
        return input_vocab, output_vocab,chars_vocab

    @classmethod
    def load_model_states(cls, path, device):
        if device == 'cuda':
            # resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            embedder = torch.load(os.path.join(path, cls.BILSTM_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))

        else:
            # resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME),
            #                                map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME), map_location=lambda storage, loc: storage)
            embedder = torch.load(os.path.join(path, cls.BILSTM_NAME), map_location=lambda storage, loc: storage)
        return embedder,model




