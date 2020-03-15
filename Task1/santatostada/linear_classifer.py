import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    
    probs = predictions
    if len(predictions.shape) == 1:
        probs =  probs - np.max(probs)
        probs = np.exp(probs) / np.sum(np.exp(probs))
    else:
        probs = probs - np.max(probs, axis = 1).reshape(-1, 1)
        probs = np.exp(probs) / np.sum(np.exp(probs), axis = 1).reshape(-1, 1)
    return probs
    
    
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    
    if probs.ndim == 1:
        loss = - np.log(probs[target_index])
    else:
        loss = - sum(np.log(probs[x, target_index[x]]) for x in range(probs.shape[0]))
        loss /= probs.shape[0]
    
    return loss
    
    
    
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    dprediction = probs
    
    if len(predictions.shape) == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(predictions.shape[0]), target_index.flatten()] -= 1
        dprediction /= predictions.shape[0]
    
    
    
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    
    
    loss = reg_strength * np.sum(W ** 2)
    grad = reg_strength * 2 * W
    
    

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    
    
    
    loss, grad = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, grad)
    
    
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            
            for batch in batches_indices:
                loss, grad = linear_softmax(X[batch], self.W, y[batch])
                reg_loss, reg_grad = l2_regularization(self.W, reg)
                loss = loss + reg_loss
                self.W = self.W - learning_rate * (grad + reg_grad)
                loss_history.append(loss)
            
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        
        
        
        predictions = np.dot(X, self.W)
        y_pred = predictions.argmax(axis = 1)

        return y_pred



                
                                                          

            

                
