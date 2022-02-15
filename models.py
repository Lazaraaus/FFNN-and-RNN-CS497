import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Feed_Forward(nn.Module):
        """
        Class for Feed Forward NN

        Activation function: 

    """
        def __init__(self, vocab_size):
                super(Feed_Forward, self).__init__()
                self.input_layer = nn.Linear(5 * 100, 100)
                #print(f"input layer: {self.input_layer.is_cuda}")
                self.output_layer = nn.Linear(100, vocab_size)
                #print(f"output layer: {self.output_layer.is_cuda}")

        def forward(self, input):
                output_layer_input = F.relu(self.input_layer(input))
                output_layer_output = self.output_layer(output_layer_input)
                return output_layer_output


