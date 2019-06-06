from collections.abc import Iterable

def metric(*args, **kwargs):
    if len(args) >= 1 and callable(args[0]):
        return MetricDef(*args, **kwargs)
    else:
        def decor(func):
            return MetricDef(func, *args, **kwargs)
        return decor

# mode = standard(output, target) universal(output, data)
class MetricDef:
    def __init__(self, func, name=None, labels=[], reduction='mean',
                 calling_convention='standard'):
        self._func = func
        self._name = name if name is not None else func.__name__
        assert isinstance(labels, Iterable), "labels should be a list, tuple or set"
        self._labels = set(labels)
        self._reduction = reduction
        self._calling_convention = calling_convention
        
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
    
    @property
    def calling_convention(self):
        return self._calling_convention
    

    