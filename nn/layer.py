import numpy as np


class Layer():

    def __init__(self, n_neur, act_f, n_conn=None):
        self.act_f = act_f
        self.n_conn = n_conn
        self.n_neur = n_neur
        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = None if n_conn is None else np.random.rand(
            n_conn, n_neur) * 2 - 1
