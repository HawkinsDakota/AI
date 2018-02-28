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

# plotting imports
import matplotlib.pyplot as plt

# iteration imports
import itertools

# machine learning evaluation imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class NeuralNet(object):

    def __init__(self, input_nodes, output_nodes, hidden_layer_nodes=[],
                 learning_rate=0.05, w_lambda=0, bias_nodes=True, verbose=True):
        """
        Generic Neural Network class wiht sigmoid activated nodes.

        Parameters:
            input_nodes (int): the number of input nodes in the network
            output_nodes (int): the number of output nodes in the network.
            hidden_layer_nodes (list, int): the number of nodes to
               include in each hidden layer in the network. Disregard if no
               hidden layers are present.
            learning_rate (float): learning rate for back propogation algorithm.
                Default is 0.05.
            w_lambda (float): parameter used in L2-Regularization during
                training. Lambda = 0 will result in no regularization.
            bias_nodes (boolean): whether to include bias nodes in network
                construction. Default is True.
            verbose (boolean): whether to plot MSE after fit. Default is True.
        
        Instance Variables:

        """
        # list indicating size transitions between each layer
        # (e.g. |input nodes| - > |output nodes|)
        shape_transitions = []
        try:
            self.bias_nodes = bool(bias_nodes)
        except:
            raise ValueError("Expected boolean for `bias_nodes")

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

        # save number of layers
        self.n_layers = len(shape_transitions)


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
            self.weights.append(np.ones((dim + self.bias_nodes,
                                shape_transitions[i + 1])))

        try:
            self.verbose = bool(verbose)  # whether to plot
        except:
            raise IOError("Expected boolean for `verbose`.")
        
        # lambda for L2-regularization
        try:
            self.w_lambda = float(0)
        except:
            raise IOError("Expected number for `w_lambda`.")
        

    def __check_input_dimensions(self, dim):
        """Ensure proper input dimensionality."""
        if dim != self.n_input:
            raise ValueError("Expected data with dim={}.".format(self.n_input))

    def __check_output_dimensions(self, dim):
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
        self.__check_input_dimensions(data.shape[1])
        labels = np.zeros(data.shape[0])
        for row in range(data.shape[0]):
            layer_outputs = self.forward(data[row, :])
            labels[row] = self.label_from_outputs(layer_outputs)

        return labels

    @staticmethod
    def label_from_outputs(layer_outputs):
        """Return predicted label returned by the network."""
        return np.argmax(layer_outputs[-1])


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
            None, unless `verbose` is True. Then returns average Mean-Square
                Error at each epoch.
        """
        self.__check_input_dimensions(data.shape[1])
        self.__check_output_dimensions(len(set(labels)))
        self.__check_label_values(labels)
        # average Mean-Squared Error for verbose plotting
        avg_mse = []
        # iterate through training epochs
        for epoch in range(epochs):
            mse = []  # Mean-Squared Error for verbose plotting.
            # iterate through samples
            for i in range(data.shape[0]):
                # create one-hot vector for cost calculation
                sample_one_hot = np.zeros(self.n_output)
                sample_one_hot[int(labels[i])] = 1
                # perform forward pass through network
                layer_outputs = self.forward(data[i, :])
                cost = self.compute_cost(layer_outputs[-1], sample_one_hot)
                self.backward(cost, layer_outputs)

                if self.verbose:
                    mse.append(self.calculate_mse(cost))

            if self.verbose:
                avg_mse.append(np.mean(mse))

        
        if self.verbose:
            plt.plot(range(epochs), avg_mse)
            plt.ylabel('Mean-Squared Error')
            plt.xlabel('Epoch')
            plt.show()
            return avg_mse
            
        return None

    @staticmethod
    def calculate_mse(cost):
        """Calculate Mean-Squared Error from prediction costs."""
        return np.sum(cost**2)

    @staticmethod
    def compute_cost(predicted, known):
        """
        Compute the cost of predicted labels from a forward pass in the network.

        Arguments:
            predicted (numpy.ndarray): A (k x 1) float array where each number
                is output generated from a forward pass through the network.

            known (numpy.ndarray): A one-hot (k x 1) integer array where all
                values are zero except for the index corresponding to the known
                class.
        
        Returns:
            (numpy.ndarray): A (k x 1) vector where each entry is the cost
                associated with its predicted assignment
        """
        if predicted.shape != known.shape:
            raise ValueError("Expected equal dimension between predicted and \
known vectors.")

        return known - predicted

    def forward(self, sample_data):
        """
        Perform a forward pass through the network. All nodes coded as sigmoid
        functions.

        Arguments:
            sample_data (numpy.ndarray): An (1 x m) data matrix where n is the
                number of samples and m is the number of input nodes provided
                in the model.

        Returns:
            (list, numpy.ndarray): A list containing outputs generated at each
                layer of the network. The first entry of the list is
                `sample_data`, while the last entry should be used for class
                prediction.
        """
        self.__check_input_dimensions(sample_data.shape[0])
        # instantiate layer 0 input to sample data
        node_input = sample_data
        if self.bias_nodes:
            node_input = np.hstack((node_input, 1))
        layer_outputs = [node_input]
        # iterate through layers
        for i, W in enumerate(self.weights):
            # calculate dot products (w'x)
            dot_products = np.dot(W.T, node_input)
            # apply sigmoid function to each dot product
            node_input = 1.0 / (1 + np.exp(-1 * dot_products))
            if self.bias_nodes and i != len(self.weights) - 1:
                node_input = np.hstack((node_input, 1))
            layer_outputs.append(node_input)

        # return outputs generated at each layer
        return layer_outputs

def backward(cost, layer_outputs):
    """
    Perform back propogration to update weights for the network.

    Arguments:
        cost (numpy.ndarray): an (n x 1) float array where each number is
            mean squared error produced by a forward pass through the
            network for each sample.

        layer_outputs(list, numpy.ndarray): output from `forward()`. List
            of node outputs at each layer of the network. 

    Returns:
        None
    """
    previous_beta = cost
    new_weights = [each.copy() for each in self.weights]
    # iterate from n to 0 layer outputs -> update weights and calculate
    # beta for next iteration.
    for idx in range(self.n_layers - 1, 0, -1):
        # update weights using previously calculated beta values
        node_output = layer_outputs[idx]
        if self.bias_nodes and (idx != self.n_layers - 1):
            node_output = node_output[0:-1]
        output_scalars = node_output * (1 - node_output) * previous_beta
        # some way to vectorize this?
        for i, o in enumerate(output_scalars):
            # calculate gradient direction term
            delta_w = self.alpha * o * layer_outputs[idx - 1]

            # calculate L2 regularization term
            last_node = self.weights[idx - 1].shape[0]

            # do not regularize bias node
            if self.bias_nodes:
                last_node = self.weights[idx - 1].shape[0] - 1

            scaled_weight = (1 - self.w_lambda * self.alpha) *\
                                self.weights[idx - 1][0:last_node, i]

            if self.bias_nodes:
                scaled_weight = np.hstack((scaled_weight, 0))
            
            new_weights[idx - 1][:, i] = scaled_weight + delta_w
        
        # calculate new betas -- remove bias factor node in weight matrix as
        # node doesn't trace back to other nodes
        W = self.weights[idx - 1]
        if self.bias_nodes:
            W = self.weights[idx - 1][0:-1, :]
        previous_beta = np.dot(W, output_scalars)
        
    self.weights = new_weights
    return None


def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.

    Modified from Lab3 -- written by CS640 TA

    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels

    returns:
        None
    """
    x1_range = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.01)
    x2_range = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.01)
    x1_array, x2_array = np.meshgrid(x1_range, x2_range)
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, linewidths=1,
                edgecolors='black', alpha=0.75)
    plt.show()


def train_and_test_fit(model, data, labels, percent=0.75, epochs=100):
    """
    Split data into test and training sets and measure performance.

    Arguments:
        model (NeuralNet): model to be trained to be trained.
        data (numpy.ndarray): (n x m) dataset to train model on where n is the
            number of samples, and m is the number of features. m should
            correspond to the number of input dimensions in `model`.
        labels (numpy.ndarray): (n x 1) numpy.ndarray containing integer labels
            for each sample in `data`.
        percent (float): percent of data to keep as test.
        epochs (integer): number of epochs
    
    Returns:
        (NeuralNet): trained model
    """
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        train_size=percent)
    model.fit(x_train, y_train, epochs)
    plot_decision_boundary(model, x_train, y_train)
    y_predicted = model.predict(x_test)
    cf_matrix = confusion_matrix(y_test, y_predicted)
    plot_confusion_matrix(cf_matrix, list(set(y_test)), normalize=True)



def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def run_experiments():
    """Run experiments as outlined in Homework 1."""
    # linear data
    linear_data = np.loadtxt("DATA/LinearX.csv", delimiter=',')
    linear_labels = np.loadtxt("DATA/LinearY.csv", delimiter=',')

    # non-linear data
    non_linear_data = np.loadtxt("DATA/NonlinearX.csv", delimiter=',')
    non_linear_labels = np.loadtxt("DATA/NonlinearY.csv", delimiter=',')

    # digit data 
    digit_train_data = np.loadtxt("DATA/Digit_X_train.csv", delimiter=',')
    digit_train_labels = np.loadtxt("DATA/Digit_y_train.csv", delimiter=',')

    digit_test_data = np.loadtxt("DATA/Digit_X_test.csv", delimiter=',')
    digit_test_labels = np.loadtxt("DATA/Digit_y_test.csv", delimiter=',')


    # 1.) 2-layer NN with linear data
    nn1 = NeuralNet(2, 2, verbose=False)
    train_and_test_fit(nn1, linear_data, linear_labels, epochs=500)

    # 2.) 2-layer NN with non-linear data
    nn2 = NeuralNet(2, 2, verbose=False)
    train_and_test_fit(nn2, non_linear_data, non_linear_labels, epochs=500)

    # 3.) 3-layer NN with 2 nodes, both datasets
    nn3 = NeuralNet(2, 2, hidden_layer_nodes=[3], verbose=True)
    nn4 = NeuralNet(2, 2, hidden_layer_nodes=[2], verbose=True)
    train_and_test_fit(nn3, linear_data, linear_labels, epochs=1500)
    train_and_test_fit(nn4, non_linear_data, non_linear_labels, epochs=500)



          



    

