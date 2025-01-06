"""
Train the passed model given the data and the batcher and save the best to disk.
"""
from __future__ import print_function
import os
import logging
import time, copy
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import transformers

from . import predict_utils as pu
from . import data_utils as du
import wandb
from src.learning.train_utils import *

LOG_EVERY = 1

class GenericTrainer:
    save_function = generic_save_function

    def __init__(self, model, batcher, model_path, train_hparams,
                 early_stop=True, verbose=True, dev_score='loss'):
        """
        A generic trainer class that defines the training procedure. Trainers
        for other models should subclass this and define the data that the models
        being trained consume.
        :param model: pytorch model.
        :param batcher: a model_utils.Batcher class.
        :param model_path: string; directory to which model should get saved.
        :param early_stop: boolean;
        :param verbose: boolean;
        :param dev_score: string; {'loss'/'f1'} How dev set evaluation should be done.
        # train_hparams dict elements.
        :param train_size: int; number of training examples.
        :param dev_size: int; number of dev examples.
        :param batch_size: int; number of examples per batch.
        :param accumulated_batch_size: int; number of examples to accumulate gradients
            for in smaller batch size before computing the gradient. If this is not present
            in the dictionary or is smaller than batch_size then assume no gradient
            accumulation.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param es_check_every: int; check some metric on the dev set every check_every iterations.
        :param lr_decay_method: string; {'exponential', 'warmuplin', 'warmupcosine'}
        :param decay_lr_by: float; decay the learning rate exponentially by the following
            factor.
        :param num_warmup_steps: int; number of steps for which to do warm up.
        :param decay_lr_every: int; decay learning rate every few iterations.

        """
        # Book keeping
        self.dev_score = dev_score
        self.verbose = verbose
        self.es_check_every = train_hparams['es_check_every']
        self.num_dev = train_hparams['dev_size']
        self.batch_size = train_hparams['batch_size']
        self.num_epochs = train_hparams['num_epochs']
        self.accumulate_gradients,self.accumulated_batch_size ,self.update_params_every = validate_accumulate_gradients(train_hparams)

        if self.accumulate_gradients:
            logging.info(f'Accumulating gradients for: {self.accumulated_batch_size}; updating params every: {self.update_params_every}; with batch size: {self.batch_size}')

        self.num_train, self.num_batches, self.total_iters = calculate_num_batches(train_hparams, self.num_epochs)
        self.model_path = model_path  # Save model and checkpoints.
        self.iteration = 0

        # Model, batcher and the data.
        self.model = model
        self.batcher = batcher
        self.time_per_batch, self.time_per_dev_pass = 0, 0
        # Different trainer classes can add this based on the data that the model they are training needs.
        self.train_fnames = []
        self.dev_fnames = {}

        # Optimizer args.
        self.early_stop = early_stop
        self.update_rule = train_hparams['update_rule']
        self.learning_rate = train_hparams['learning_rate']
        self.optimizer = get_optimizer(self.model, self.update_rule, self.learning_rate)

        # Reduce the learning rate every few iterations.
        self.lr_decay_method = train_hparams['lr_decay_method']
        self.decay_lr_every = train_hparams['decay_lr_every']
        self.log_every = LOG_EVERY
        # self.wandb_callback = WandbCallback(log_every=5)
        self.scheduler =get_lr_scheduler(self.optimizer, self.lr_decay_method, train_hparams, self.num_batches)

        # Train statistics.
        self.loss_history = defaultdict(list)
        self.loss_checked_iters = []
        self.dev_score_history = []
        self.dev_checked_iters = []

        # Every subclass needs to set this.
        self.loss_function_cal = GenericTrainer.compute_loss

    def train(self):
        """
        Make num_epoch passes through the training set and train the model.
        :return:
        """
        # Pick the model with the least loss.
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_score = -np.inf

        total_time_per_batch = 0
        total_time_per_dev = 0
        train_start = time.time()
        logging.info('num_train: {:d}; num_dev: {:d}'.format(
            self.num_train, self.num_dev))
        logging.info('Training {:d} epochs, {:d} iterations'.
                     format(self.num_epochs, self.total_iters))
        for epoch, ex_fnames in zip(range(self.num_epochs), self.train_fnames):
            # Initialize batcher. Shuffle one time before the start of every
            # epoch.
            epoch_batcher = self.batcher(ex_fnames=ex_fnames,
                                         num_examples=self.num_train,
                                         batch_size=self.batch_size)
            # Get the next training batch.
            iters_start = time.time()
            for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
                self.model.train()
                batch_start = time.time()
                with torch.amp.autocast('cuda:0', dtype=torch.bfloat16):
                    if self.accumulate_gradients:
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters
                        objective.backward()
                        if (self.iteration + 1) % self.update_params_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    else:
                        # Clear all gradient buffers.
                        self.optimizer.zero_grad()
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters.
                        objective.backward()
                        # Step in the direction of the gradient.
                        self.optimizer.step()

                if self.iteration % self.log_every == 0:
                    wandb.log({
                        "train_loss": objective.item(),
                        "epoch": epoch,
                        "iteration": self.iteration,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                    # Save every loss component separately.
                    loss_str = []
                    for key in ret_dict:
                        if torch.cuda.is_available():
                            loss_comp = float(ret_dict[key].data.cpu().numpy())
                        else:
                            loss_comp = float(ret_dict[key].data.numpy())
                        self.loss_history[key].append(loss_comp)
                        loss_str.append('{:s}: {:.4f}'.format(key, loss_comp))
                    self.loss_checked_iters.append(self.iteration)
                    if self.verbose:
                        log_str = 'Epoch: {:d}; Iteration: {:d}/{:d}; '.format(epoch, self.iteration, self.total_iters)
                        logging.info(log_str + '; '.join(loss_str))
                elif self.verbose:
                    logging.info('Epoch: {:d}; Iteration: {:d}/{:d}'.
                                 format(epoch, self.iteration, self.total_iters))
                # The decay_lr_every doesnt need to be a multiple of self.log_every
                if self.iteration > 0 and self.iteration % self.decay_lr_every == 0:
                    self.scheduler.step()
                    # logging.info('Decayed learning rates: {}'.
                    #              format([g['lr'] for g in self.optimizer.param_groups]))
                batch_end = time.time()
                total_time_per_batch += batch_end - batch_start
                # Check every few iterations how you're doing on the dev set.
                if self.iteration % self.es_check_every == 0 and self.iteration != 0 and self.early_stop:
                    # Save the loss at this point too.
                    for key in ret_dict:
                        if torch.cuda.is_available():
                            loss_comp = float(ret_dict[key].data.cpu().numpy())
                        else:
                            loss_comp = float(ret_dict[key].data.numpy())
                        self.loss_history[key].append(loss_comp)
                    self.loss_checked_iters.append(self.iteration)
                    # Switch to eval model and check loss on dev set.
                    self.model.eval()
                    dev_start = time.time()
                    # Returns the dev F1.
                    if self.dev_score == 'f1':
                        dev_score = pu.batched_dev_scores(
                            model=self.model, batcher=self.batcher, batch_size=self.batch_size,
                            ex_fnames=self.dev_fnames, num_examples=self.num_dev)
                    elif self.dev_score == 'loss':
                        dev_score = -1.0 * pu.batched_loss(
                            model=self.model, batcher=self.batcher, batch_size=self.batch_size,
                            ex_fnames=self.dev_fnames, num_examples=self.num_dev,
                            loss_helper=self.loss_function_cal)
                        wandb.log({"dev_loss": dev_score})

                    dev_end = time.time()
                    total_time_per_dev += dev_end - dev_start
                    self.dev_score_history.append(dev_score)
                    self.dev_checked_iters.append(self.iteration)
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        # Deep copy so you're not just getting a reference.
                        best_params = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_iter = self.iteration
                        everything = (epoch, self.iteration, self.total_iters, dev_score)
                        wandb.log({"best_dev_loss": dev_score, "best_epoch": epoch, "best_iteration": self.iteration})

                        if self.verbose:
                            logging.info('Current best model; Epoch {:d}; '
                                         'Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                        # self.save_function(model=self.model, save_path=self.model_path, model_suffix='cur_best')
                        wandb.save(os.path.join(self.model_path, 'best_model.pt'))
                    else:
                        everything = (epoch, self.iteration, self.total_iters, dev_score)
                        if self.verbose:
                            logging.info('Epoch {:d}; Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

                    # for key in self.loss_history:
                    #     du.plot_train_hist(self.loss_history[key], self.loss_checked_iters,
                    #                        fig_path=self.model_path, ylabel=f"{key}-{self.iteration}")
                    # du.plot_train_hist(self.dev_score_history, self.dev_checked_iters,
                    #                    fig_path=self.model_path, ylabel=f'Dev-set Score {self.iteration}')
                self.iteration += 1
            epoch_time = time.time() - iters_start
            logging.info('Epoch {:d} time: {:.4f}s'.format(epoch, epoch_time))
            logging.info('\n')

        # Say how long things took.
        train_time = time.time() - train_start
        logging.info('Training time: {:.4f}s'.format(train_time))
        if self.total_iters > 0:
            self.time_per_batch = float(total_time_per_batch) / self.total_iters
        else:
            self.time_per_batch = 0.0
        logging.info('Time per batch: {:.4f}s'.format(self.time_per_batch))
        if self.early_stop and self.dev_score_history:
            if len(self.dev_score_history) > 0:
                self.time_per_dev_pass = float(total_time_per_dev) / len(self.dev_score_history)
            else:
                self.time_per_dev_pass = 0
            logging.info('Time per dev pass: {:4f}s'.format(self.time_per_dev_pass))

        # Save the learnt model: save both the final model and the best model.
        # https://stackoverflow.com/a/43819235/3262406
        self.save_function(model=self.model, save_path=self.model_path, model_suffix='final')
        logging.info('Best model; Epoch {:d}; Iteration {:d}; Dev loss: {:.4f}'
                     .format(best_epoch, best_iter, best_dev_score))
        self.model.load_state_dict(best_params)
        self.save_function(model=self.model, save_path=self.model_path, model_suffix='best')

        # Plot training time stats.
        for key in self.loss_history:
            du.plot_train_hist(self.loss_history[key], self.loss_checked_iters,
                               fig_path=self.model_path, ylabel=key)
        du.plot_train_hist(self.dev_score_history, self.dev_checked_iters,
                           fig_path=self.model_path, ylabel='Dev-set Score')

    @staticmethod
    def compute_loss(loss_components):
        """
        Models will return dict with different loss components, use this and compute batch loss.
        :param loss_components: dict('str': Variable)
        :return:
        """
        raise NotImplementedError


class BasicRankingTrainer(GenericTrainer):
    def __init__(self, model, batcher, model_path, data_path,
                 train_hparams, early_stop=True, verbose=True, dev_score='loss'):
        """
        Trainer for any model returning a ranking loss. Uses everything from the
        generic trainer but needs specification of how the loss components
        should be put together.
        :param data_path: string; directory with all the int mapped data.
        """
        GenericTrainer.__init__(self,
                                model=model,
                                batcher=batcher,
                                model_path=model_path,
                                train_hparams=train_hparams,
                                early_stop=early_stop,
                                verbose=verbose,
                                dev_score=dev_score)
        # Expect the presence of a directory with as many shuffled copies of the
        # dataset as there are epochs and a negative examples file.
        self.train_fnames = []
        # Expect these to be there for the case of using diff kinds of training data for the same
        # model; hard negatives models, different alignment models and so on.
        if 'train_suffix' in train_hparams:
            suffix = train_hparams['train_suffix']
            train_basename = f'train-{suffix}'
            dev_basename = f'dev-{suffix}'
        else:
            train_basename = 'train'
            dev_basename = 'dev'
        for i in range(self.num_epochs):
            # Each run contains a copy of shuffled data for itself.
            ex_fname = {
                'pos_ex_fname': os.path.join(data_path, 'shuffled_data', f'{train_basename}-{i}.jsonl'),
                # 'pos_ex_fname': os.path.join(data_path, '{:s}.jsonl'.format(train_basename)),
            }
            self.train_fnames.append(ex_fname)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, f'{dev_basename}.jsonl'),
        }
        # Every subclass needs to set this.
        self.loss_function_cal = BasicRankingTrainer.compute_loss


    @staticmethod
    def compute_loss(loss_components):
        """
        Simply add loss components.
        :param loss_components: dict('rankl': rank loss value)
        :return: Variable.
        """
        return loss_components['rankl']


def conditional_log(logger, process_rank, message):
    """
    Helper to log only from one process when using DDP.
    -- logger is entirely unused. Dint seem to work when used in conjuncton with cometml.
    """
    if process_rank == 0:
        logger.info(message)

class GenericTrainerDDP:
    # If this isnt set outside of here it crashes with: "got multiple values for argument"
    # todo: Look into who this happens --low-pri.s
    save_function = generic_save_function_ddp

    def __init__(self, logger, process_rank, num_gpus, model, batcher, model_path, train_hparams,
                 early_stop=True, verbose=True, dev_score='loss'):
        """
        A generic trainer class that defines the training procedure. Trainers
        for other models should subclass this and define the data that the models
        being trained consume.
        :param logger: a logger to write logs with.
        :param process_rank: int; which process this is.
        :param num_gpus: int; how many gpus are being used to train.
        :param model: pytorch model.
        :param batcher: a model_utils.Batcher class.
        :param model_path: string; directory to which model should get saved.
        :param early_stop: boolean;
        :param verbose: boolean;
        :param dev_score: string; {'loss'/'f1'} How dev set evaluation should be done.
        # train_hparams dict elements.
        :param train_size: int; number of training examples.
        :param dev_size: int; number of dev examples.
        :param batch_size: int; number of examples per batch.
        :param accumulated_batch_size: int; number of examples to accumulate gradients
            for in smaller batch size before computing the gradient. If this is not present
            in the dictionary or is smaller than batch_size then assume no gradient
            accumulation.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param es_check_every: int; check some metric on the dev set every check_every iterations.
        :param lr_decay_method: string; {'exponential', 'warmuplin', 'warmupcosine'}
        :param decay_lr_by: float; decay the learning rate exponentially by the following
            factor.
        :param num_warmup_steps: int; number of steps for which to do warm up.
        :param decay_lr_every: int; decay learning rate every few iterations.

        """
        # Book keeping
        self.process_rank = process_rank
        self.logger = logger
        self.dev_score = dev_score
        self.verbose = verbose
        self.num_epochs = train_hparams['num_epochs']

        self.es_check_every = train_hparams['es_check_every'] // num_gpus
        self.num_train, self.num_batches, self.total_iters = calculate_num_batches(train_hparams, self.num_epochs,
                                                                                   num_gpus)
        self.num_dev = train_hparams['dev_size']
        self.batch_size = train_hparams['batch_size']
        self.num_epochs = train_hparams['num_epochs']
        self.accumulate_gradients, self.accumulated_batch_size, self.update_params_every = validate_accumulate_gradients(
            train_hparams)
        if self.accumulate_gradients:
            conditional_log(self.logger, self.process_rank,
                            f'Accumulating gradients for: {self.accumulated_batch_size}; updating params every: {self.update_params_every}; with batch size: {self.batch_size}')
        self.model_path = model_path  # Save model and checkpoints.
        self.iteration = 0

        # Model, batcher and the data.
        self.model = model
        self.batcher = batcher
        self.time_per_batch = 0
        self.time_per_dev_pass = 0

        # Different trainer classes can add this based on the data that the model they are training needs.
        self.train_fnames = []
        self.dev_fnames = {}

        # Optimizer args.
        self.early_stop = early_stop
        self.update_rule = train_hparams['update_rule']
        self.learning_rate = train_hparams['learning_rate']

        # Initialize optimizer.
        self.optimizer = get_optimizer(self.model, self.update_rule, self.learning_rate)


        # Reduce the learning rate every few iterations.
        self.lr_decay_method = train_hparams['lr_decay_method']
        self.decay_lr_every = train_hparams['decay_lr_every']
        self.scheduler = get_lr_scheduler(self.optimizer, self.lr_decay_method, train_hparams, self.num_batches,
                                          num_gpus)
        self.log_every = LOG_EVERY

        # Train statistics.
        self.loss_history = defaultdict(list)
        self.loss_checked_iters = []
        self.dev_score_history = []
        self.dev_checked_iters = []

        # Every subclass needs to set this.
        self.loss_function_cal = GenericTrainer.compute_loss


    def dev_step(self):
        dev_start = time.time()

        self.model.eval()
        # Using the module as it is for eval:
        # https://discuss.pytorch.org/t/distributeddataparallel-barrier-doesnt-work-as-expected-during-evaluation/99867/11
        dev_score = -1.0 * pu.batched_loss_ddp(
            model=self.model.module, batcher=self.batcher, batch_size=self.batch_size,
            ex_fnames=self.dev_fnames, num_examples=self.num_dev,
            loss_helper=self.loss_function_cal, logger=self.logger)
        dev_end = time.time()
        dev_time = dev_end - dev_start
        return dev_score, dev_time

    def train_step(self,epoch, epoch_batcher, total_time_per_batch, total_time_per_dev, best_dev_score, best_params, best_epoch, best_iter):
        for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
            # print(f"Got batch: {os.getpid()}")
            self.model.train()
            batch_start = time.time()
            # Impelemented according to:
            # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-
            # manually-to-zero-in-pytorch/4903/20
            # With my implementation a final batch update may not happen sometimes but shrug.
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not self.accumulate_gradients:
                    self.optimizer.zero_grad()
                ret_dict = self.model.forward(batch_dict=batch_dict)
                objective = self.compute_loss(loss_components=ret_dict)
                objective.backward()
                if self.accumulate_gradients:
                    if (self.iteration + 1) % self.update_params_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()

                # The decay_lr_every doesnt need to be a multiple of self.log_every
                if self.iteration > 0 and self.iteration % self.decay_lr_every == 0:
                    self.scheduler.step()


            # Log metrics
            if self.iteration % self.log_every == 0:
                if self.process_rank == 0:
                    wandb.log({
                        "train_loss": objective.item(),
                        "epoch": epoch,
                        "iteration": self.iteration,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                # # Save every loss component separately.
                loss_str = []
                for key in ret_dict:
                    if torch.cuda.is_available():
                        loss_comp = float(ret_dict[key].data.cpu().numpy())
                    else:
                        loss_comp = float(ret_dict[key].data.numpy())
                    self.loss_history[key].append(loss_comp)
                    loss_str.append(f'{key}: {loss_comp}')
                self.loss_checked_iters.append(self.iteration)
                if self.verbose:
                    log_str = (f'Epoch: {epoch}; Iteration: {self.iteration}/{self.total_iters}; '
                               + '; '.join(loss_str))
                    conditional_log(self.logger, self.process_rank, log_str)
            elif self.verbose:
                log_str = f'Epoch: {epoch}; Iteration: {self.iteration}/{self.total_iters}'
                conditional_log(self.logger, self.process_rank, log_str)

            batch_end = time.time()
            total_time_per_batch += batch_end - batch_start
            # Check every few iterations how you're doing on the dev set.
            if self.iteration % self.es_check_every == 0 and self.iteration != 0 and self.early_stop and self.process_rank == 0:

                # Save the loss at this point too.
                for key in ret_dict:
                    if torch.cuda.is_available():
                        loss_comp = float(ret_dict[key].data.cpu().numpy())
                    else:
                        loss_comp = float(ret_dict[key].data.numpy())
                    self.loss_history[key].append(loss_comp)
                self.loss_checked_iters.append(self.iteration)
                # Switch to eval model and check loss on dev set.
                dev_score, dev_time = self.dev_step()
                total_time_per_dev += dev_time
                wandb.log({"dev_loss": dev_score})
                self.dev_score_history.append(dev_score)
                self.dev_checked_iters.append(self.iteration)
                if dev_score > best_dev_score:
                    best_dev_score = dev_score
                    # Deep copy so you're not just getting a reference.
                    best_params = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    best_iter = self.iteration
                    everything = (epoch, self.iteration, self.total_iters, dev_score)
                    if self.verbose:
                        self.logger.info('Current best model; Epoch {:d}; '
                                         'Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                    wandb.log({"best_dev_loss": dev_score, "best_epoch": epoch, "best_iteration": self.iteration})
                    self.save_function(model=self.model, save_path=self.model_path, model_suffix='cur_best')

                else:
                    everything = (epoch, self.iteration, self.total_iters, dev_score)
                    if self.verbose:
                        self.logger.info('Epoch {:d}; Iteration {:d}/{:d}; Dev score: {:.4f}'
                                         .format(*everything))
            dist.barrier()
            self.iteration += 1
        return best_params, best_epoch, best_iter

    def train(self):
        """
        Trains the model over a specified number of epochs and iterations, performs optimization, evaluates
        on the development dataset, and manages early stopping if applicable. The method additionally tracks
        and logs relevant training metrics and saves both the final and best-performing models.

        :param self: Represents the instance of the class containing this method.

        :raises AnyErrorType: Raised when there are underlying issues during training (e.g., batch creation,
            model optimization). Replace `AnyErrorType` with specific exceptions if determinable.

        :param self.model: The model being trained.
        :param self.model_path: Path where the trained models (final and best) will be stored.
        :param self.num_epochs: Number of epochs for the training process.
        :param self.total_iters: Total iterations performed during training.
        :param self.batch_size: Batch size used for data sampling during training and evaluation.
        :param self.num_train: Number of training examples available.
        :param self.num_dev: Number of development examples available.
        :param self.dev_score_history: Holds the score values from past evaluations on the dev dataset, used for early stopping.
        """
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_score = -np.inf
        total_time_per_batch = 0
        total_time_per_dev = 0
        conditional_log(self.logger, self.process_rank,
                        f'num_train: {self.num_train}; num_dev: {self.num_dev}')
        conditional_log(self.logger, self.process_rank,
                        f'Training {self.num_epochs} epochs, {self.total_iters} iterations')
        # print(f"pid: {os.getpid()}, checkpoint: {self.train_fnames}")

        train_start = time.time()

        for epoch, ex_fnames in zip(range(self.num_epochs), self.train_fnames):
            # Initialize batcher. Shuffle one time before the start of every epoch.
            # conditional_log(self.logger, self.process_rank,message=f"{ex_fnames}. creating batcher..")
            epoch_batcher = self.batcher(ex_fnames=ex_fnames,
                                         num_examples=self.num_train,
                                         batch_size=self.batch_size)
            # conditional_log(self.logger, self.process_rank,message=f"batcher created"
            iters_start = time.time()
            best_params, best_epoch, best_iter = self.train_step(epoch=epoch,
                                                                 epoch_batcher=epoch_batcher,
                                                                 total_time_per_batch=total_time_per_batch,
                                                                 total_time_per_dev=total_time_per_dev,
                                                                 best_dev_score=best_dev_score,
                                                                 best_params=best_params,
                                                                 best_epoch=best_epoch,
                                                                 best_iter=best_iter)
            epoch_time = time.time() - iters_start
            conditional_log(self.logger, self.process_rank, f'Epoch {epoch} time: {epoch_time:.4f}s')

        train_time = time.time() - train_start
        # Log time stats
        conditional_log(self.logger, self.process_rank, f'Training time: {train_time}s')
        self.time_per_batch = float(total_time_per_batch) / self.total_iters if self.total_iters > 0 else 0.0
        conditional_log(self.logger, self.process_rank, f'Time per batch: {self.time_per_batch:.4f}s')

        if self.early_stop and self.dev_score_history:
            self.time_per_dev_pass = float(total_time_per_dev) / len(self.dev_score_history) if len(self.dev_score_history) > 0 else 0
            conditional_log(self.logger, self.process_rank, f'Time per dev pass: {self.time_per_dev_pass:4f}s')

        # Save the learnt model: save both the final model and the best model.
        # https://stackoverflow.com/a/43819235/3262406
        if self.process_rank == 0:
            self.save_function(model=self.model, save_path=self.model_path, model_suffix='final')
            self.model.load_state_dict(best_params)
            self.save_function(model=self.model, save_path=self.model_path, model_suffix='best')
            conditional_log(self.logger, self.process_rank, f'Best model; Epoch {best_epoch}; Iteration {best_iter}; Dev loss: {best_dev_score:.4f}')


    @staticmethod
    def compute_loss(loss_components):
        """
        Models will return dict with different loss components, use this and compute batch loss.
        :param loss_components: dict('str': Variable)
        :return:
        """
        raise NotImplementedError


class BasicRankingTrainerDDP(GenericTrainerDDP):
    def __init__(self, logger, process_rank, num_gpus, model, batcher, model_path, data_path,
                 train_hparams, early_stop=True, verbose=True, dev_score='loss'):
        """
        Trainer for any model returning a ranking loss. Uses everything from the
        generic trainer but needs specification of how the loss components
        should be put together.
        :param data_path: string; directory with all the int mapped data.
        """
        GenericTrainerDDP.__init__(self, logger=logger, process_rank=process_rank, num_gpus=num_gpus,
                                   model=model, batcher=batcher, model_path=model_path,
                                   train_hparams=train_hparams, early_stop=early_stop, verbose=verbose,
                                   dev_score=dev_score)
        # Expect the presence of a directory with as many shuffled copies of the dataset as there are epochs and a negative examples file.
        self.train_fnames = []

        # Expect these to be there for the case of using diff kinds of training data for the same
        # model; hard negatives models, different alignment models and so on.
        if 'train_suffix' in train_hparams:
            suffix = train_hparams['train_suffix']
            train_basename = 'train-{:s}'.format(suffix)
            dev_basename = 'dev-{:s}'.format(suffix)
        else:
            train_basename = 'train'
            dev_basename = 'dev'
        for i in range(self.num_epochs):
            # Each run contains a copy of shuffled data for itself
            # Each process gets a part of the data to consume in training.
            ex_fname = {
                'pos_ex_fname': os.path.join(data_path, 'shuffled_data', f'{train_basename}-{process_rank}-{i}.jsonl'),
            }
            self.train_fnames.append(ex_fname)
            # print(self.train_fnames)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, '{:s}.jsonl'.format(dev_basename)),
        }
        # The split command in bash is asked to make exactly equal sized splits with the remainder in a final file which is unused

        # Every subclass needs to set this.
        self.loss_function_cal = BasicRankingTrainer.compute_loss
    @staticmethod
    def compute_loss(loss_components):
        """
        Simply add loss components.
        :param loss_components: dict('rankl': rank loss value)
        :return: Variable.
        """
        return loss_components['rankl']

def calculate_num_batches(train_hparams,num_epochs, num_gpus=1):
    """
    Calculate the number of samples per GPU, batches per epoch, and total iterations.

    :param train_hparams: Hyperparameters for the training process containing:
        - 'batch_size': Size of each training batch.
        - 'train_size': Total number of training examples.
    :type train_hparams: dict
    :param num_epochs: Total number of epochs for training.
    :type num_epochs: int
    :param num_gpus: Number of GPUs used for training (default: 1).
    :type num_gpus: int, optional

    :return: A tuple containing:
        - num_samples_per_gpu: Number of training examples per GPU.
        - batches_per_epoch: Number of batches in one epoch.
        - total_iterations: Total iterations over all epochs.
    :rtype: tuple
    """
    batch_size = train_hparams['batch_size']
    train_size = train_hparams['train_size']
    num_samples_per_gpu = train_size // num_gpus
    batches_per_epoch = int(np.ceil(num_samples_per_gpu / batch_size)) if num_samples_per_gpu > batch_size else 1
    total_iterations = num_epochs * batches_per_epoch
    return num_samples_per_gpu, batches_per_epoch, total_iterations

def get_optimizer(model, update_rule, learning_rate):
    # Initialize optimizer.
    if update_rule == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer
    elif update_rule == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
        return optimizer
    else:
        raise ValueError(f'Unknown update rule: {update_rule}')

def get_lr_scheduler(optimizer, lr_decay_method,train_hparams, num_batches, num_gpus=1):
    """
    Creates a learning rate scheduler based on the specified decay method.

    :param optimizer: Optimizer for which the scheduler is created.
    :param lr_decay_method: Decay method ('exponential', 'warmuplin', 'warmupcosine').
    :param train_hparams: Dictionary with relevant hyperparameters:
        - 'decay_lr_by' for 'exponential'.
        - 'num_warmup_steps' and 'num_epochs' for warmup methods.
    :param num_batches: Number of batches per epoch.
    :param num_gpus: Number of GPUs used in training (default: 1).
    :return: Configured learning rate scheduler.
    :rtype: torch.optim.lr_scheduler._LRScheduler or transformers.get_scheduler
    """
    if lr_decay_method == 'exponential':
        decay_lr_by = train_hparams['decay_lr_by']
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=decay_lr_by)
        return scheduler
    elif lr_decay_method == 'warmuplin':
        num_warmup_steps = train_hparams['num_warmup_steps'] // num_gpus
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 # Total number of training batches.
                                                                 num_training_steps=train_hparams['num_epochs'] * num_batches)
        return scheduler
    elif lr_decay_method == 'warmupcosine':
        num_warmup_steps = train_hparams['num_warmup_steps'] // num_gpus
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 num_training_steps=train_hparams['num_epochs'] * num_batches)
        return scheduler
    else:
        raise ValueError(f'Unknown lr_decay_method: {train_hparams['lr_decay_method']}')

def validate_accumulate_gradients(train_hparams):
    """
       Validates and calculates parameters for gradient accumulation.

       Gradient accumulation allows effective batch size enlargement by accumulating gradients
       over multiple mini-batches before updating model parameters.

       :param train_hparams: Dict containing training hyperparameters:
           - 'batch_size': Required, positive integer.
           - 'accumulated_batch_size': Optional, positive integer greater than and a multiple of
             'batch_size'. If 0 or -1, gradient accumulation is disabled.
       :type train_hparams: dict

       :return: Tuple (accumulate_gradients, accumulated_batch_size, update_params_every):
           - accumulate_gradients: Whether gradient accumulation is enabled (bool).
           - accumulated_batch_size: Effective accumulated batch size (int).
           - update_params_every: Steps before updating parameters (int).
       :rtype: tuple
    """
    batch_size = train_hparams['batch_size']
    accumulated_batch_size = train_hparams.get("accumulated_batch_size", -1)
    if accumulated_batch_size > 0:
        # It should be bigger and an exact multiple of the batch size.
        if accumulated_batch_size <= batch_size:
            raise ValueError("'accumulated_batch_size' must be greater than 'batch_size'.")
        if accumulated_batch_size % batch_size != 0:
            raise ValueError("'accumulated_batch_size' must be an exact multiple of 'batch_size'.")
        update_params_every = accumulated_batch_size // batch_size
        accumulate_gradients = True
    else:
        accumulate_gradients = False
        update_params_every = 1

    return accumulate_gradients, accumulated_batch_size, update_params_every