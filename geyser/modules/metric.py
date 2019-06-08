from collections.abc import Iterable

def metric(*args, **kwargs):
    if len(args) >= 1 and callable(args[0]):
        return MetricDef(*args, **kwargs)
    else:
        def decor(func):
            return MetricDef(func, *args, **kwargs)
        return decor

class MetricDef:
    def __init__(self, func, name=None, labels=[], reduction='mean'):
        self._func = func
        self._name = name if name is not None else func.__name__
        assert isinstance(labels, (list, tuple, set)), "labels should be a list, tuple or set"
        self._labels = set(labels)
        assert isinstance(reduction, (str, list, tuple)), "reduction should be a str, list or tuple"
        self._reduction = reduction
        
    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    @property
    def name(self):
        return self._name
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def reduction(self):
        return self._reduction
    

class MetricsCollector:
    def __init__(self, metrics_defs):
        self.__metrics_defs = metrics_defs
        self.__metric_name2def = {}
        for metric_def in metrics_defs:
            assert metric_def.name not in self.__metric_name2def, "the metric name should be unique"
            self.__metric_name2def[metric_def.name] = metric_def
        self.clear()
    
    def collect(self, name, value):
        assert name in self.__metrics_series, 'invalid metric name'
        self.__metrics_series[name].append(value)
    
    def reduce(self):
        res = {}
        for name in self.__metrics_series:
            name = self.__metrics_series[name]
            reduction = self.__metric_name2def[name].reduction
            res[name] = self.__reduce(name, reduction)
        return res

    def clear(self):
        self.__metrics_series = { metric_def.name : [] for metric_def in metrics_defs }

    @staticmethod
    def __reduce(metric_series, reduction):
        if isinstance(reduction, (list, tuple)):
            res = {}
            for item in reduction:
                assert isinstance(item, str), 'reduction item should be a string'
                res[item] = Trial.__reduce(metric_series, item)
            return res
        elif reduction == 'mean':
            return torch.mean(torch.tensor(metric_series, dtype=torch.float)).item()
        elif reduction == 'std':
            return torch.std(torch.tensor(metric_series, dtype=torch.float)).item()
        elif reduction == 'random':
            return random.choice(metric_series)
        elif reduction == 'max':
            return torch.max(torch.tensor(metric_series)).item()
        elif reduction == 'min':
            return torch.min(torch.tensor(metric_series)).item()
        elif reduction == 'median':
            return torch.median(torch.tensor(metric_series)).item()
        elif reduction == 'sum':
            return torch.sum(torch.tensor(metric_series)).item()
