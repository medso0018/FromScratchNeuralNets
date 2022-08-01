import numpy as np


class Layer():

    def __init__(self, n_neur, act_f, layer_type='output_layer', n_conn=None):
        self.act_f = act_f
        self.n_conn = n_conn
        self.n_neur = n_neur
        self.type = layer_type
        self.b = np.random.rand(n_neur) * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1
