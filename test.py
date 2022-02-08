import torch	# Torch
import toch.nn as nn # NN Models
import torch.nn.functional as F # Non-Linearities

x = torch.rand(5,3)
print(x)

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]


word_to_ix = {}
for sent, _ in data + test_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)

NUM_LABELS = 2

class BoWClassifier(nn.Module):
	def __init__(self, num_labels, vocab_size):
		super(BoWClassifier, self).__init__()

		# Define the input/output matrices for this NN
		self.linear = nn.Linear(vocab_size, num_labels)
	
	# Define the forward pass function
	def forward(self, bow_vec):
		# Pass through linear layer, then pass through softmax
		return F.relu(self.linear(bow_vec), dim=1)
