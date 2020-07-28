class NullPolicy(object):
    def __init__(self, _):
        pass

    @staticmethod
    def get_action(_):
        return None
