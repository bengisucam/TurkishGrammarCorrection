from __future__ import division
import logging
import os
import random
import datetime as dt
import time
from datetime import datetime

import torch
import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): Experiments Directory to store details of the Experiments,
            by default it makes a folder in the current directory to store the details (default: `Experiments`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for Experiments, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """

    def __init__(self, expt_dir='Experiments', save_name="Experiment01", loss=NLLLoss(), batch_size=64,
                 random_seed=None, print_every=100, early_stop_threshold=4, checkpoint_every=2):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None

        self.print_every = print_every
        self.early_stop = early_stop_threshold
        self.checkpoint_every = checkpoint_every
        self.save_name = save_name

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(os.path.join(self.expt_dir, save_name)):
            os.makedirs(os.path.join(self.expt_dir, save_name))
        self.batch_size = batch_size

    def _train_batch(self, input_variable, input_lengths, target_variable, model, bilstm, teacher_forcing_ratio):
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio,
                                                       bilstm=bilstm)

        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            step_o = step_output.contiguous().view(batch_size, -1)
            loss.eval_batch(step_o, target_variable[:, step + 1])

        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data,char_vocab, model, bilstm, n_epochs, start_epoch, start_step,
                       dev_data=None, test_data=None, teacher_forcing_ratio=0, deviceName='cuda'):

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch
        device = torch.device(deviceName)
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0

        epoch_dev_loss = 0
        epoch_dev_min_loss = 1111111
        kill_signal_count = 0

        for epoch in range(start_epoch, n_epochs + 1):
            epoch_time = time.time()
            if 0 < epoch_dev_loss < epoch_dev_min_loss:
                epoch_dev_min_loss = epoch_dev_loss
                kill_signal_count = 0
            if epoch_dev_loss > epoch_dev_min_loss:
                kill_signal_count += 1
                if kill_signal_count >= self.early_stop:
                    logging.info("- terminating due to early stopping(Threshold : {})".format(self.early_stop))
                    break
            logging.info("- epoch: %d, total steps: %d" % (epoch, total_steps - step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            bilstm.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1
                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)
                loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model, bilstm,
                                         teacher_forcing_ratio)
                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every - 1:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    logging.info(
                        ('\t- progress: {}%  train {}: {}  step: {}  time: {} '.format(int((step / total_steps) * 100),
                                                                                       self.loss.name,
                                                                                       print_loss_avg,
                                                                                       step,
                                                                                       datetime.now().strftime(
                                                                                           "%H:%M:%S"))))

                # Checkpoint

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "- finished epoch %d: with train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, bilstm, dev_data, device)
                epoch_dev_loss = dev_loss
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
                bilstm.train(True)

            else:
                self.optimizer.update(epoch_loss_avg, epoch)
            log_msg += f' in {dt.timedelta(seconds=time.time() - epoch_time)}'
            logging.info(log_msg)

            if epoch % self.checkpoint_every == 0 or step == total_steps:
                self._save(model, bilstm, epoch, step, data, test_data=test_data, deviceName=deviceName,char_vocab=char_vocab)
        return model

    def _save(self, model, bilstm, epoch, step, data,char_vocab, test_data, deviceName):
        path = Checkpoint(model=model,
                          bilstm=bilstm,
                          optimizer=self.optimizer,
                          epoch=epoch, step=step,
                          input_vocab=data.fields[seq2seq.src_field_name].vocab,
                          output_vocab=data.fields[seq2seq.tgt_field_name].vocab,
                          char_vocab=char_vocab).save(self.expt_dir,
                                                                                       f'{self.save_name}\\{epoch}')
        if test_data is not None:
            test_loss, test_acc = self.evaluator.evaluate(model, bilstm, test_data, torch.device(deviceName))
            logging.info(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
        return path

    def _save_final(self, model, bilstm, epoch, step, data,char_vocab):
        path = Checkpoint(model=model,
                          bilstm=bilstm,
                          optimizer=self.optimizer,
                          epoch=epoch, step=step,
                          input_vocab=data.fields[seq2seq.src_field_name].vocab,
                          output_vocab=data.fields[seq2seq.tgt_field_name].vocab,
                          char_vocab=char_vocab).save(self.expt_dir,
                                                                                       f'{self.save_name}\\Final')
        return path

    def train(self, model, bilstm, data,char_vocab, num_epochs=5, dev_data=None, test_data=None,
              optimizer=None, teacher_forcing_ratio=0, deviceName='cuda', resume=False):
        """ Run training for a given model.
        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
            :param bilstm:
            :param resume:
        """

        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path, 'cuda')
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer
            bilstm = resume_checkpoint.bilstm
            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        # print("- optimizer: %s, scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data,char_vocab, model, bilstm, num_epochs,
                            start_epoch, step, dev_data=dev_data, test_data=test_data,
                            teacher_forcing_ratio=teacher_forcing_ratio, deviceName=deviceName)
        path = self._save_final(model, bilstm, num_epochs, step, data,char_vocab)
        if test_data is not None:
            test_loss, test_acc = self.evaluator.evaluate(model, bilstm, test_data, torch.device(deviceName))
            logging.info(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
        return model, path
