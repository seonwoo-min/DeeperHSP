# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.evaluate import compute_metrics
from src.utils import Print


class Trainer():
    """ train / eval helper class """
    def __init__(self, model):
        self.model = model
        self.optim = None
        self.scheduler = None
        self.class_weight = None

        # initialize logging parameters
        self.train_flag = False
        self.epoch = 0.0
        self.best_loss = None
        self.logger_train = Logger()
        self.logger_eval  = Logger()

    def train(self, batch, device):
        # training of the model
        batch = set_device(batch, device)

        self.model.train()
        self.optim.zero_grad()
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = get_loss(outputs, labels, self.class_weight)
        loss.backward()
        self.optim.step()

        # logging
        self.logger_train.update(len(outputs), loss.item())
        self.logger_train.keep(F.softmax(outputs, dim=1), labels)

    def evaluate(self, batch, device):
        # evaluation of the model
        batch = set_device(batch, device)

        self.model.eval()
        with torch.no_grad():
            inputs, labels = batch
            outputs = self.model(inputs)
            loss = get_loss(outputs, labels, self.class_weight).item()

        # logging
        self.logger_eval.update(len(outputs), loss)
        self.logger_eval.keep(F.softmax(outputs, dim=1), labels)

    def scheduler_step(self):
        # scheduler_step
        self.scheduler.step()

    def save_model(self, save_prefix):
        # save state_dicts to checkpoint """
        if save_prefix is None: return
        elif not os.path.exists(save_prefix + "/checkpoints/"):
            os.makedirs(save_prefix + "/checkpoints/", exist_ok=True)

        state = self.model.state_dict()
        torch.save(state, save_prefix + "/checkpoints/%d.pt" % self.epoch)

    def save_outputs(self, save_prefix):
        # save state_dicts to checkpoint """
        if save_prefix is None: return
        self.logger_eval.evaluate(train=False)
        np.save(save_prefix + "/Y.npy", self.logger_eval.labels)
        np.save(save_prefix + "/P.npy", self.logger_eval.outputs)

    def load_model(self, checkpoint, output):
        # load state_dicts from checkpoint """
        if checkpoint is None: return
        Print('loading a model state_dict from the checkpoint', output)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v
        self.model.load_state_dict(state_dict, strict=False)

    def set_class_weight(self, labels, run_cfg):
        if not run_cfg.class_weight: return
        num_classes = torch.max(labels) + 1
        class_weight = torch.zeros(num_classes).float()

        for i in range(num_classes):
            class_weight[i] = torch.sum(labels == i)
        max_num = torch.max(class_weight)
        for i in range(num_classes):
            class_weight[i] = torch.sqrt(max_num / class_weight[i])

        self.class_weight = class_weight

    def set_device(self, device):
        # set gpu configurations
        self.model = self.model.to(device)
        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(device)

    def set_optim_scheduler(self, run_cfg, params):
        # set optim and scheduler for training
        optim, scheduler = get_optim_scheduler(run_cfg, params)
        self.train_flag = True
        self.optim = optim
        self.scheduler = scheduler

    def headline(self, output):
        # get a headline for logging
        headline = []
        if self.train_flag:
            headline += ["ep", "idx"]
            headline += self.logger_train.get_headline(train=True)
        else:
            headline += ["idx"]
            headline += self.logger_eval.get_headline(train=False)

        Print("\t".join(headline), output)

    def log(self, idx, output):
        # logging
        log = []
        if self.train_flag:
            self.logger_train.evaluate(train=True)
            log += ["%03d" % self.epoch, idx]
            log += self.logger_train.log
        else:
            self.logger_eval.evaluate(train=False)
            log += [idx]
            log += self.logger_eval.log

        Print("\t".join(log), output)
        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_train.reset()
        self.logger_eval.reset()


class Logger():
    """ Logger class """
    def __init__(self):
        self.total = 0.0
        self.loss = 0.0
        self.outputs = []
        self.labels = []
        self.log = []

    def update(self, total, loss):
        # update logger for current mini-batch
        self.total += total
        self.loss += loss * total

    def keep(self, outputs, labels):
        # keep labels and outputs for future computations
        self.outputs.append(outputs.cpu().detach().numpy())
        self.labels.append(labels.cpu().detach().numpy())

    def get_loss(self):
        # get current averaged loss
        loss = self.loss / self.total
        return loss

    def get_headline(self, train=False):
        # get headline
        if train: headline = ["loss"]
        else:     headline = ["acc", "f1", "pr", "re", "sp", "mcc", "auroc", "aupr"]

        return headline

    def evaluate(self, train=False):
        # compute evaluation metrics
        self.aggregate()
        if train: evaluations = [self.get_loss()]
        else:     evaluations = [*compute_metrics(self.labels, self.outputs)]
        self.log = ["%.4f" % eval for e, eval in enumerate(evaluations)]

    def aggregate(self):
        # aggregate kept labels and outputs
        if isinstance(self.outputs, list) and len(self.outputs) > 0:
            self.outputs = np.concatenate(self.outputs, axis=0)
        if isinstance(self.labels, list) and len(self.labels) > 0:
            self.labels = np.concatenate(self.labels, axis=0)

    def reset(self):
        # reset logger
        self.total = 0.0
        self.loss = 0.0
        self.outputs = []
        self.labels = []
        self.log = []


def get_optim_scheduler(cfg, params):
    """ configure optim and scheduler """
    optim = torch.optim.Adam(params, lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.2)

    return optim, scheduler


def get_loss(outputs, labels, class_weight):
    """ get cross entropy loss """
    loss = F.cross_entropy(outputs, labels, weight=class_weight)

    return loss


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch

