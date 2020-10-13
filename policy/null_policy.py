"""
null policy

input a null policy the system will perform the default actuate control (SUMO)

"""


class NullPolicy(object):
    def __init__(self, _):
        pass

    @staticmethod
    def get_action(_):
        return None
