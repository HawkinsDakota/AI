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

    def __init__(input_nodes=0, output_nodes=0, hidden_layer_nodes=[],
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

        shape_transitions = []

        try:
            self.n_input = int(input_nodes)
        except:
            raise IOError("Expected integer for `input_nodes`")
        shape_transitions.append(self.n_input)

        for x in hidden_layer_nodes:
            try:
                hidden_nodes = int(x)
            except:
                raise IOError("Expected integer")
            shape_transitions.append(hidden_nodes)
            
        
        try:
            self.n_output = int(output_nodes)
        except:
            raise IOError("Expected integer for `output_nodes`")
        shape_transitions.append(self.n_output)



        try:
            self.alpha = float(learning_rate)
        except:
            raise IOError("Expected float for `learning_rate`.")

        self.weights = []
        for each in shape_transitions:
            

