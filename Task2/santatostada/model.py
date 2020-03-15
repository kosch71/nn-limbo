import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for ix in self.params():
            self.params()[ix] = np.zeros_like(self.params()[ix].value)

        out1 = self.ReLU1.forward(self.fc1.forward(X))
        out2 = self.fc2.forward(out1)

        loss, grad = softmax_with_cross_entropy(out2, y)

        d_out2 = self.fc2.backward(grad)
        self.fc1.backward(self.ReLU1.backward(d_out2))

        for param_ix in self.params():
            l2_loss, l2_grad = l2_regularization(self.params()[param_ix].value, self.reg)
            loss += l2_loss
            self.params()[param_ix].grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        out1 = self.ReLU1.forward(self.fc1.forward(X))
        out2 = self.fc2.forward(out1)
        pred = out2.argmax(axis=1)
        
        return pred

    def params(self):
        result = {'W1': self.fc1.W, 'B1': self.fc1.B, 'W2': self.fc2.W, 'B2': self.fc2.B}

        return result
