import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.__initialize_weights()

    def __initialize_weights(self):
        self.weights = 0.01 * np.random.randn(self.n_inputs, self.n_neurons)
        self.bias = np.zeros((1, self.n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.bias = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.inputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def __init__(self):
        pass

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # We subtract the largest neuron to avoid dead nueron problem
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def __init__(self):
        pass
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
    
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods