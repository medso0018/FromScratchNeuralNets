from nn.loss import loss_function
import pandas as pd


class NeuralNetwork():

    def __init__(self, loss_f='MSE'):
        self.loss_f = loss_function(loss_f=loss_f)
        self.layers = []

    # add layer to the network
    def add_layer(self, layer):
        if len(self.layers) != 0:
            layer.connect(self.layers[-1])
        self.layers.append(layer)

    # train
    def fit(self, X, Y, epochs=1000, lr=0.05):
        loss = []
        for _ in range(epochs):
            error = 0
            for x, y in zip(X, Y):

                # forward
                out = x
                for layer in self.layers[1:]:
                    out = layer.forward(out)

                # error
                error += self.loss_f.loss(y, out)

                # backward
                grad = self.loss_f.loss_prime(y, out)
                for layer in self.layers[::-1][:-1]:
                    grad = layer.backward(grad, lr)

            loss.append(error / len(Y))
        return loss

    # prediction
    def predict(self, X):
        output = []
        for x in X:
            out = x
            for layer in self.layers[1:]:
                out = layer.forward(out)
            output.append(out)
        return output

    # model's summary
    def summary(self):
        summary = {
            'Type': [],
            'Activation Function': [],
            'Input Shape': [],
            'Output Shape': []
        }

        for layer in self.layers:
            summary['Type'].append(layer.__class__.__name__)
            summary['Activation Function'].append(layer.act_f_name)
            summary['Input Shape'].append(layer.input_shape)
            summary['Output Shape'].append(layer.output_shape)

        return pd.DataFrame(summary)
