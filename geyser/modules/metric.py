def metric(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return Metric(args[0], **kwargs)
    else:
        def metric_(func):
            return Metric(func, *args, **kwargs)
        return metric_
    
class Metric:
    def __init__(self, func, labels=[]):
        self.func = func
        self.name = func.__name__
        assert(isinstance(labels, list) or isinstance(labels, tuple))            
        self.labels = labels
        
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, '__metric_cache'):
            obj.__metric_cache = {}
        obj.__metric_cache.setdefault(self.name, self.func(obj))
        return obj.__metric_cache[self.name]
    
    def __set__(self, obj, value):
        raise AttributeError("can't set metric")
    
    def __delete__(self, obj):
        raise AttributeError("can't delete metric")

    
class Metrics:
    def __init__(self):
        subclass = self.__class__
        self._attrs = [attr for attr in dir(subclass) if isinstance(getattr(subclass, attr), Metric)]

    def select(self, selected_label = None):
        if selected_label is None:
            return { attr : getattr(self, attr) for attr in self._attrs }
        subclass = self.__class__
        return { attr : getattr(self, attr) for attr in self._attrs if selected_label in getattr(subclass, attr).labels }
    
    def __repr__(self):
        return str(self.select())
    