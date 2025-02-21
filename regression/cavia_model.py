import torch
import torch.nn.functional as F
from torch import nn


class CaviaModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_in,
                 n_out,
                 num_context_params,
                 n_hidden,
                 device
                 ):
        super(CaviaModel, self).__init__()

        self.device = device

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        
        # KL: first layer takes in input features + context params as input and outputs n_hidden[0] neurons 
        self.fc_layers.append(nn.Linear(n_in + num_context_params, n_hidden[0]))

        # KL: creating fully connected layers by iterating through n_hidden 
        for k in range(len(n_hidden) - 1):
            self.fc_layers.append(nn.Linear(n_hidden[k], n_hidden[k + 1]))
        self.fc_layers.append(nn.Linear(n_hidden[-1], n_out))

        # context parameters (note that these are *not* registered parameters of the model!)
        # KL: this difference is important because MAML updates ALL model parameters during training together. 
        # KL: CAVIA updates only the context_params in inner loop and the rest of the model in outer loop SEPARATELY. 
        self.num_context_params = num_context_params
        self.context_params = None
        self.reset_context_params()

    def reset_context_params(self):
        self.context_params = torch.zeros(self.num_context_params).to(self.device)
        self.context_params.requires_grad = True

    def forward(self, x):

        # KL: concatenate input with context parameters
        x = torch.cat((x, self.context_params.expand(x.shape[0], -1)), dim=1)

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y
