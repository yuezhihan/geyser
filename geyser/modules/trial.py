import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
from collections.abc import Iterable
from .utils import pack_arguments
from .metric import MetricsManager
from ..VERSION import VERSION as _VERSION
from enum import Enum

class TrialStatus:
    NOTRUNNING = 0
    RUNNING = 1
    STOPPING = 2

    def __init__(self):
        self._n_epoch = 0
        self._n_iteration = 0
        self._running = NOTRUNNING
    
    @property
    def n_epoch(self):
        return self._n_epoch
    
    @property
    def n_iteration(self):
        return self._n_iteration

    @property
    def running(self):
        return self._running

class Trial:
    VERSION = _VERSION

    def __init__(self, model, metrics_defs, train_loader,
                 valid_loader = None, valid_every = None,
                 event_handlers = [],
                 data_transform_func = Trial.default_transform_data):
        assert isinstance(model, nn.Module), "model should be a Module"
        self.__model = model

        assert isinstance(metrics_defs, (list, tuple)), "metrics_defs should be a list or tuple"
        self.__metrics_defs = list(metrics_defs)
        self.__loss = None
        for metric_def in metrics_defs:
            if 'loss' in metric_def.labels:
                self.__loss = metric_def.name
                break
        assert self.__loss is not None, "there must be a metric labeled 'loss'"

        # self.__metric_label2defs = defaultdict(set)
        # for metric_def in metrics_defs:
        #     for label in metric_def.labels:
        #         self.__metric_label2defs[label].add(metric_def)

        self.__event_handlers = event_handlers
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_every = valid_every
        self.test_loader = test_loader
        self.__transform_data = data_transform_func
        self.__status = TrialStatus()
    
    @property
    def model(self):
        return self.__model
    
    @property
    def metrics_defs(self):
        return self.__metrics_defs

    @property
    def train_loader(self):
        return self.__train_loader
    @train_loader.setter
    def train_loader(self, val):
        assert isinstance(val, DataLoader), "train_loader should be an instance of DataLoader"
        self.__train_loader = val

    @property
    def valid_loader(self):
        return self.__valid_loader
    @valid_loader.setter
    def valid_loader(self, val):
        assert val is None or isinstance(val, DataLoader), "valid_loader should be an instance of DataLoader"
        self.__valid_loader = val
    
    # @property
    # def test_loader(self):
    #     return self.__test_loader
    # @test_loader.setter
    # def test_loader(self, val):
    #     assert val is None or isinstance(val, DataLoader), "test_loader should be an instance of DataLoader"
    #     self.__test_loader = val

    @property
    def status(self):
        return self.__status

    @property
    def transform_data(self):
        return self.__transform_data

    @staticmethod
    def default_transform_data(data):
        input, target = data
        return pack_arguments(input), pack_arguments(target)

    def stop(self):
        self.status._running = TrialStatus.STOPPING

    def __run_metrics(self, output, target, metrics_series):
        res = {}
        for metric_def in self.__metrics_defs:
            name = metric_def.name
            metric_value = metric_def(output, *target[0], **target[1])
            metrics_series.collect(name, metric_value)
            res[name] = metric_value
        return res
 
    def evaluate(self, data_loader, metrics_series = None):
        org_training = self.__model.training
        self.__model.eval()
        if metrics_series is None:
            metrics_series = MetricsSeries()
        if isinstance(data_loader, DataLoader):
            with torch.no_grad():
                for data in data_loader:
                    input, target = self.transform_data(data)
                    output = self.__model(*input[0], **input[1])
                    self.__run_metrics(output, target, metrics_series)
                res = metrics_series.reduce()
        else: # data_loader is a single example
            with torch.no_grad():
                input, target = self.transform_data(data_loader)
                output = self.__model(*input[0], **input[1])
                res = self.__run_metrics(output, target, metrics_series)
        self.__model.train(org_training)
        return res

    def run_for(self, max_epochs, optimizer):
        return self.run(max_epochs - self.status.n_epoch, optimizer)

    def run(self, num_epochs, optimizer):
        assert num_epochs >= 0, 'the number of epochs should be greater than or equal to zero'
        org_training = self.__model.training
        self.__model.train()
        
        for _ in range(num_epochs):
            metrics_series = MetricsSeries()
            for data in self.train_loader:
                optimizer.zero_grad()
                input, target = self.transform_data(data)
                output = self.__model(*input[0], **input[1])
                metrics = self.__run_metrics(output, target, metrics_series)
                loss = metrics[self.__loss]
                loss.backward()
                optimizer.step()

                if self.status.n_iteration % self.valid_every == 0:
                    self.evaluate(self.valid_loader)
                self.status.n_iteration += 1

            overall_metrics = metrics_series.reduce()
            self.status._n_epoch += 1
            
        self.__model.train(org_training)

    # def test(self):
    #     return self.evaluate(self.test_loader)
    
    def load(self):
        return self.__model.load_state_dict(dict)

    def save(self):
        return self.__model.state_dict()

__all__ = ['Trial']
