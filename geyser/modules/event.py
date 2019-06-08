# @Events.iteration_started

# 支持class注册和function注册两种

class Events:
    @staticmethod
    def iteration_started(func):
        pass

    @staticmethod
    def iteration_completed(func):
        pass

    @staticmethod
    def epoch_started(func):
        pass

    @staticmethod
    def epoch_completed(func):
        pass
    