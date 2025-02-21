"""
Neural network models for the regression experiments
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class MamlModel(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_weights,
                 num_context_params,
                 device
                 ):
        """
        :param n_inputs:            the number of inputs to the network
        :param n_outputs:           the number of outputs of the network
        :param n_weights:           for each hidden layer the number of weights
        :param num_context_params:  number of additional inputs (trained together with rest)
        """
        super(MamlModel, self).__init__()

        # initialise lists for biases and fully connected layers
        self.weights = []
        self.biases = []

        # KL: list to store number of nodes per layer
        # KL: n_weights = [40, 40] for sinusoidal task 
        # KL: n_outputs nodes for last layer 
        self.nodes_per_layer = n_weights + [n_outputs]

        # additional biases
        self.task_context = torch.zeros(num_context_params).to(device)
        self.task_context.requires_grad = True

        # KL: prev_n_weight keeps track of input size for each layer 
        # KL: first layer takes in both input features and task context params
        prev_n_weight = n_inputs + num_context_params

        for i in range(len(self.nodes_per_layer)):
            # KL: creates a weight matrix for layer (prev_n_weight x layer size)
            w = torch.Tensor(size=(prev_n_weight, self.nodes_per_layer[i])).to(device)
            w.requires_grad = True
            
            # KL: creates a bias vector for layer 
            self.weights.append(w)
            b = torch.Tensor(size=[self.nodes_per_layer[i]]).to(device)
            b.requires_grad = True
            self.biases.append(b)

            # KL: update prev_n_weight
            prev_n_weight = self.nodes_per_layer[i]

        self._reset_parameters()

    def _reset_parameters(self):
        # KL: initialise weights and biases 
        for i in range(len(self.nodes_per_layer)):
            stdv = 1. / math.sqrt(self.nodes_per_layer[i])
            self.weights[i].data.uniform_(-stdv, stdv)
            self.biases[i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # KL: concatenates task_context to input x, if task_context is not empty, expand it to match batch size 
        if len(self.task_context) != 0:
            x = torch.cat((x, self.task_context.expand(x.shape[0], -1)), dim=1)
        else:
            x = torch.cat((x, self.task_context))

        # pass thru hidden layers with ReLU and compute output w/o activation function 
        for i in range(len(self.weights) - 1):
            x = F.relu(F.linear(x, self.weights[i].t(), self.biases[i]))
        y = F.linear(x, self.weights[-1].t(), self.biases[-1])

        return y
