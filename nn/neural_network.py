from nn.layer import Layer
from nn.activation_functions import act_function
from nn.loss_functions import loss_function


class NeuralNetwork():

    def __init__(self, loss_f='mse'):
        self.loss_f = loss_function(loss_f=loss_f)
        self.layers = []

    # add layer to our network
    def add_layer(self, n_neur=5, n_conn=None, act_f='linear'):
        _n_conn = n_conn if n_conn is not None else self.layers[-1].n_neur
        if len(self.layers) != 0:
            self.layers[-1].type = 'hidden_layer'
        self.layers.append(
            Layer(
                n_neur,
                n_conn=_n_conn,
                act_f=act_function(act_f)
            )
        )

    # forward propagation
    def __forward_propagation(self, X):
        output = [(None, X)]
        for layer in self.layers:
            z = output[-1][1] @ layer.W + layer.b
            a = layer.act_f[0](z)
            output.append((z, a))
        return output

    # backward propagation
    def __back_propagation(self, y, output):
        delta = []
        for l in reversed(range(0, len(self.layers))):
            print(l)
            _, a = output[l+1]
            if self.layers[l].type == 'output_layer':
                delta.insert(
                    0, self.loss_f[1](y, a) * self.layers[l].act_f[1](a))
            else:
                delta.insert(
                    0, delta[0] @ self.layers[l+1].W.T * self.layers[l].act_f[1](a))

        return delta

    # train
    def fit(self, X_train, y_train, epoch=1000, lr=0.01):
        for _ in range(epoch):
            output = self.__forward_propagation(X_train)
            deltas = self.__back_propagation(y_train, output)
            print([a.shape for a in deltas])
            break
        return self

    # prediction
    def predict(self, X_test):
        return self.__forward_propagation(X_test)[-1][1]
