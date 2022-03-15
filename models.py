import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb

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
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
                self.embeddings.weight.requires_grad = True
                #print(f"output layer: {self.output_layer.is_cuda}")

        def forward(self, input):
                # Get Word Embeddings
                embeds = self.embeddings(input).view((-1, 5 * 100 ))
                # Compute h_t
                h_t = torch.tanh(self.input_layer(embeds))
                # Compute W_2.h_t
                logits = self.output_layer(h_t)
                # Compute Log Probs
                log_probs = F.log_softmax(logits, dim=-1)
                # Return
                return logits, log_probs

                # OG Way 
                #first_layer_input = self.input_layer(input)
                #output_layer_input = F.relu(first_layer_input)
                #pdb.set_trace()
                #output_layer_output = self.output_layer(output_layer_input)
                # Update Embeddings
                #return output_layer_output


