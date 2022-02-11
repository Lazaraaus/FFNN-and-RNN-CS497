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
		self.input_layer = nn.Linear(5 * 100, 32182) 
		self.output_layer = nn.Linear(32182, vocab_size)

	def forward(self, input):
		output_layer_input = F.relu(self.input_layer(input))
		output_layer_output = F.relu(self.output_layer(output_layer_input))
		return output_layer_output


