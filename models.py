import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Recurrent_Neural_Network(nn.Module):
        def __init__(self, vocab_size):
                super(Recurrent_Neural_Network, self).__init__()

                self.rnn = nn.RNN(100,100,1,batch_first=True)

                print("RNN was done successfully.")

                self.fc = nn.Linear(100, vocab_size)


                self.embeddings = nn.Embedding(vocab_size, 100)
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
                self.embeddings.weight.requires_grad = True

        def forward(self, input, hidden_input):
                embeds = self.embeddings(input).view((-1, 30, 100))

                output, hidden = self.rnn(embeds, hidden_input)

                out = self.fc(hidden)

                output = F.log_softmax(out, dim = -1)

                return out, hidden
                


