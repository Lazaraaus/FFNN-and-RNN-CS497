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
                print(f"The type of the input to the forward pass is: {input.dtype}")
                first_layer_input = self.input_layer(input)
                output_layer_input = F.relu(first_layer_input)
                output_layer_output = self.output_layer(output_layer_input)
                # Update Embeddings
                self.embeddings(output_layer_output.long())
                print("Done with training loop/n")
                return output_layer_output


