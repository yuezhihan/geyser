import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from collections.abc import Iterable
from .utils import pack_arguments
from collections import defaultdict

class TrainStatus:
    def __init__(self):
        self._epoch = 0
        self._metrics = {}
    
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def metric(self):
        return self._metrics
    
    @property
    def select_metrics(self):
        pass

class Trial:
    def __init__(self, model, metrics_defs, train_loader,
                 valid_loader = None, test_loader = None,
                 data_transform_func = Trial.default_transform_data):
        assert isinstance(model, nn.Module), "model should be a Module"
        self.__model = model

        assert isinstance(metrics_defs, Iterable), "metrics_defs should be a list or tuple"
        self.__metrics_defs = list(metrics_defs)

        self.__metric_name2def = {}
        for metric_def in metrics_defs:
            assert metric_def.name not in self.__metric_name2def, "the metric name should be unique"
            self.__metric_name2def[metric_def.name] = metric_def
        
        self.__metric_label2defs = defaultdict(set)
        for metric_def in metrics_defs:
            for label in metric_def.labels:
                self.__metric_label2defs[label].add(metric_def)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.__transform_data = data_transform_func
        self.__status = TrainStatus()
        self.stop()
    
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
    
    @property
    def test_loader(self):
        return self.__test_loader
    @test_loader.setter
    def test_loader(self, val):
        assert val is None or isinstance(val, DataLoader), "test_loader should be an instance of DataLoader"
        self.__test_loader = val

    @property
    def status(self):
        return self.__status

    @property
    def metrics_defs(self):
        return self.__metrics_defs

    @property
    def transform_data(self):
        return self.__transform_data

    @staticmethod
    def default_transform_data(data):
        input, target = data
        return pack_arguments(input), pack_arguments(target)

    def stop(self):
        self.__stop = True

    def __cal_metrics(self, output, data):
        metrics = {}
        for metric_def in self.__metrics_defs:
            name = metric_def.name
            _, target = self.transform_data(data)
            metrics[name] = metric_def(output, *target[0], **target[1])
        return metrics

    def evaluate(self, data_loader):
        self.__model.eval()
        metrics_series = defaultdict(lambda: None)
        index = 0
        with torch.no_grad():
            for data in data_loader:
                output = self.__model(data)
                metrics = self.__cal_metrics(output, data)
                for name in metrics:
                    self.__metric_name2def[name].reduction

                    self.__metric_reduce(self.__metric_name2def[name].reduction,
                                         metrics[name],
                                         metrics_accumulator[name],
                                         index)
                index += 1
        self.__model.train()
        return metrics

    def run(self, num_epoch, optimizer, callbacks):
        for _ in range(num_epoch):
            self.epoch += 1
            for X, Y in self.train_loader:
                Y_hat, train_metrics = self.model_exec.optimize(X, Y)
            
            Y_hat, valid_metrics = self.test()
            
            self.liveloss.update({
                ** { k : v for k, v in train_metrics.indicators().items() },
                ** { "val_" + k : v for k, v in valid_metrics.indicators().items() }
            })
            if(self.epoch % 10 == 1):
                self.liveloss.draw()
            print(f"{self.epoch} down")
            if(valid_metrics.loss() < self.min_loss):
                self.save_checkpoint(self.trainer_name + "_min_loss")
                self.min_loss = valid_metrics.loss()
    
    def test(self):
        return self.evaluate(self.test_loader)
    
    # def reset_logs(self):
    #     self.epoch = 0
    #     self.min_loss = float('inf')
    #     self.liveloss = PlotLosses()
    #     self.checkpoints = {}
    
    # def load_min_loss(self):
    #     return self.load_checkpoint(self.trainer_name + "_min_loss")
    
    # def save_checkpoint(self, name = None):
    #     if(name is None):
    #         name = self.trainer_name
    #     self.checkpoints[name] = {
    #         "trainer": self.trainer_name,
    #         "epoch": self.epoch,
    #         "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     }
    #     checkpoint = {
    #         "model": self.model_exec.state_dict(),
    #         "epoch": self.epoch,
    #         "min_loss": self.min_loss,
    #         "liveloss": self.liveloss,
    #         "checkpoints": self.checkpoints,
    #         "rng_state": torch.get_rng_state()
    #     }
    #     torch.save(checkpoint, 'models/checkpoint_%s.pt' % name)
    
    # def load_checkpoint(self, name = None):
    #     if(name is None):
    #         name = self.trainer_name
    #     checkpoint = torch.load('models/checkpoint_%s.pt' % name)
    #     self.model_exec.load_state_dict(checkpoint["model"])
    #     self.epoch = checkpoint["epoch"]
    #     self.min_loss = checkpoint["min_loss"]
    #     self.liveloss = checkpoint["liveloss"]
    #     self.checkpoints = checkpoint["checkpoints"]
    #     torch.set_rng_state(checkpoint["rng_state"])


# class RNNStocksExec:  # 管理模型和超参数，负责模型的一切操作
#     def __init__(self, model, max_grad = None):
#         self.model = model
#         self.max_grad = max_grad
#         self.set_optim("Adam")
        
#     def set_optim(self, optim_name, lr=0.007, weight_decay = 0.):
#         if optim_name == 'Adam':
#             self.optimizer = optim.Adam([
#                 {'params': find_params(self.model, 'weight'), 'weight_decay': weight_decay},
#                 {'params': find_params(self.model, 'bias')},
#             ], lr = lr)
#         elif optim_name == 'SGD':
#             self.optimizer = optim.SGD([
#                {'params': find_params(self.model, 'weight'), 'weight_decay': weight_decay},
#                {'params': find_params(self.model, 'bias')},
#             ], lr = lr / 10, momentum = 0.9)  # 由于momentum，实际lr=arg_lr*10
#         else: assert False, "unknown optim name!"
        
#     def optimize(self, X, Y):
#         self.model.train()
#         self.optimizer.zero_grad()
#         Y_hat = self.model(X, Y.size(1))
#         metrics = Metrics(Y_hat, Y, X)
#         metrics.loss().backward()
#         if self.max_grad is not None:
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
#         self.optimizer.step()
#         return Y_hat, metrics
     
#     def test(self, X, Y):
#         self.model.eval()
#         with torch.no_grad():
#             Y_hat = self.model(X, Y.size(1))
#             metrics = Metrics(Y_hat, Y, X)
#         self.model.train()
#         return Y_hat, metrics
    
#     def state_dict(self):
#         # 还可以保存超参数
#         return self.model.state_dict()
    
#     def load_state_dict(self, dict):
#         return self.model.load_state_dict(dict)


__all__ = ['Trial']
