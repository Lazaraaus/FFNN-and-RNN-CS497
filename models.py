import torch
import pdb
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
                log_probs = F.log_softmax(output_layer_output, dim=-1)
                #print(f"The log_probs are: {log_probs}")
                return log_probs

class Recurrent_Neural_Network(nn.Module):
        def __init__(self, vocab_size):
                super(Recurrent_Neural_Network, self).__init__()
                self.input_layer = nn.Linear(30 * 100 + 100, 100)
                self.hidden_layer_one = nn.Linear(100,500)
                self.hidden_layer_two = nn.Linear(500,100)
                self.output_layer = nn.Linear(30 * 100 + 100, 1)
                self.rnn = nn.RNN(100,100,1,batch_first=True)
                print("RNN was done successfully.")
                self.fc = nn.Linear(100, vocab_size)


                self.embeddings = nn.Embedding(vocab_size, 100)
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
                self.embeddings.weight.requires_grad = True

        def forward(self, input, hidden_input):
                embeds = self.embeddings(input).view((-1, 30, 100))
                #print("Embeds shape", embeds.shape)
                output, hidden = self.rnn(embeds, hidden_input)
                #hidden_one = self.input_layer(combined)
                #hidden_two  = self.hidden_layer_one(hidden_one)
                #hidden_final = self.hidden_layer_two(hidden_two)
                #output = self.output_layer(combined)
                #output = F.log_softmax(output, dim = -1)
                #print("Shape Output",output.shape)
                out = self.fc(hidden)
                #print("Shape out", out.shape)
                #print("Shape Hidden",hidden.shape)
                output = F.log_softmax(out, dim = -1)
                return out, hidden
                


