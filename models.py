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
                # Output Layer as Embedding  
                self.output_layer = nn.Linear(100, vocab_size)
                self.embeddings = nn.Embedding(vocab_size, 100)
                #print(f"output layer: {self.output_layer.is_cuda}")

        def forward(self, input):
                input = input #.type(torch.DoubleTensor)
                #print(f"The type of the input to the forward pass is: {input.dtype}")
                first_layer_input = self.input_layer(input)
                #self.embeddin
                #print(f"The shape of the first_layer_input is: {first_layer_input.shape}")
                output_layer_input = F.relu(first_layer_input)
                #print(f"The shape of the output_layer_input is: {output_layer_input.shape}")
                output_layer_output = self.output_layer(output_layer_input)
                # Update Embeddings
                log_probs = F.log_softmax(output_layer_output)
                #print(f"The log_probs are: {log_probs}")
                return log_probs


