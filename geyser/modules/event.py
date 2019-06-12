# @Events.iteration_started

# 支持class注册和function注册两种

class Events:
    @staticmethod
    def before_train(func):
        pass

    @staticmethod
    def after_train(func):
        pass

    @staticmethod
    def before_epoch(func):
        pass

    @staticmethod
    def after_epoch(func):
        pass
    
    @staticmethod
    def before_iterate(func):
        pass

    @staticmethod
    def after_iterate(func):
        pass

    @staticmethod
    def before_forward(func):
        pass

    @staticmethod
    def after_forward(func):
        pass

    @staticmethod
    def before_backward(func):
        pass

    @staticmethod
    def after_backward(func):
        pass

    @staticmethod
    def before_validate(func):
        pass

    @staticmethod
    def after_validate(func):
        pass

    @staticmethod
    def before_stop(func):
        pass

    @staticmethod
    def load(func):
        pass

    @staticmethod
    def save(func):
        pass
