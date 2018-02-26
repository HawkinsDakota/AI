"""
This is a python script to create a 2 or 3 layer Neural Net to perform
classification.

@author: Dakota Hawkins
@class: BU CS640
@date: February 23, 2018
@title: Homework 1
"""

# numerical imports
import numpy as np

class NeuralNet(object):

    def __init__(self, input_nodes, output_nodes, hidden_layer_nodes=[],
                 learning_rate=0.05):
        """
        Generic Neural Network class. Accepts a single hidden layer

        Parameters:
            input_nodes (int): the number of input nodes in the network
            output_nodes (int): the number of output nodes in the network.
            hidden_layer_nodes (list, int): the number of nodes to
               include in each hidden layer in the network. Disregard if no
               hidden layers are present.
            learning_rate (float): learning rate for back propogation algorithm.
                Default is 0.05. 
        
        Instance Variables:

        """
        # list indicating size transitions between each layer
        # (e.g. |input nodes| - > |output nodes|)
        shape_transitions = []

        # initiate input nodes
        try:
            self.n_input = int(input_nodes)
        except:
            raise IOError("Expected integer for `input_nodes`")
        shape_transitions.append(self.n_input)

        # initiate any hidden layers
        for x in hidden_layer_nodes:
            try:
                hidden_nodes = int(x)
            except:
                raise IOError("Expected integer")
            shape_transitions.append(hidden_nodes)
            
        # initiate output layers
        try:
            self.n_output = int(output_nodes)
        except:
            raise IOError("Expected integer for `output_nodes`")
        shape_transitions.append(self.n_output)


        # set learning rate 
        try:
            self.alpha = float(learning_rate)
        except:
            raise IOError("Expected float for `learning_rate`.")

        # initiate trasition weight matrices between each layer
        # all weights instantiated as 1
        # layer i nodes as rows, layer i + 1 nodes as columns
        self.weights = []
        for i, dim in enumerate(shape_transitions[0:-1]):
            self.weights.append(np.ones((dim, shape_transitions[i + 1])))

        self.verbose = True  # whether to plot

    def __check_input_dimensions(self, dim):
        """Ensure proper input dimensionality."""
        if dim != self.n_input:
            raise ValueError("Expected data with dim={}.".format(self.n_input))

    def __check_output_dimension(self, dim):
        """Ensure proper output dimensionality."""
        if dim != self.n_output:
            raise ValueError("Expected data with dim={}.".format(self.n_output))

    def __check_label_values(self, labels):
        label_range = range(0, self.n_output)
        for x in labels:
            try:
                int_x = int(x)
            except:
                raise ValueError("Expected numeric value for `labels` values")
            if int_x != x:
                raise ValueError("Expected integer for `labels` values")
            if int_x not in label_range:
                raise ValueError("Integer labels must be between 0-{}".format(
                                 self.n_output))
    
    def predict(self, data):
        """
        Predict class membership for proved samples.

        Arguments:
            data (numpy.ndarray): An n x m data matrix where n is the number 
                of samples and m is the number of input nodes provided in the
                model.
        
        Returns:
            (numpy.ndarray): an (n x 1) integer array where each integer
                corresponds to the predicted class for that sample. 
        """
        return 0

    def fit(self, data, labels, epochs=100):
        """
        Fit the NeuralNet to the provided labeled dataset.

        Arguments:
            data (numpy.ndarray): An (n x m) data matrix where n is the number 
                of samples and m is the number of input nodes provided in the
                model.

            labels (numpy.ndarray): An (k x 1) integer array where each integer
                corresponds to the known class for that sample.

            epochs (int): the number of training epochs.

        Returns:
            None
        """
        self.__check_input_dimensions(data.shape[1])
        self.__check_output_dimensions(len(set(labels)))
        self.__check_label_values(labels)
        # iterate through training epochs
        for epoch in range(epochs):
            # iterate through samples
            for i in range(data.shape[0]):
                # create one-hot vector for cost calculation
                sample_one_hot = np.zeros(self.n_output)
                sample_one_hot[int(labels[i])] = 1
                # perform forward pass through network
                sample_output = self.forward(data[i, :])
            
        return None

    def compute_cost(self, predicted, known):
        """
        Compute the cost of predicted labels from a forward pass in the network.

        Arguments:
            predicted (numpy.ndarray): A (k x 1) float array where each number
                is output generated from a forward pass through the network.

            known (numpy.ndarray): A (k x 1) integer array where each integer
                corresponds to the known class for each sample.
        
        Returns:
            (float): Mean-Squared Error produced by predictions.
        """

        return 0

    def forward(self, sample_data):
        """
        Perform a forward pass through the network.

        Arguments:
            sample_data (numpy.ndarray): An (1 x m) data matrix where n is the
                number of samples and m is the number of input nodes provided
                in the model.

        Returns:
            (numpy.ndarray): A (k x 1) float array where each number maps to
            class prediction.
        """
        self.__check_input_dimensions(sample_data.shape[0])
        # instantiate layer 0 input to sample data
        node_input = sample_data

        # iterate through layers
        for W in self.weights:
            # calculate dot products (w'x)
            dot_products = np.dot(W.T, node_input)
            # apply sigmoid function to each dot product
            node_input = 1.0 / (1 + np.exp(-1 * dot_products))

        # return final sigmoid output
        return node_input

    def backward(self, cost):
        """
        Perform back propogration to update weights for the network.

        Arguments:
            cost (numpy.ndarray): an (n x 1) float array where each number is
                mean squared error produced by a forward pass through the
                network for each sample.

        Returns:
            None
        """
        return None

            

