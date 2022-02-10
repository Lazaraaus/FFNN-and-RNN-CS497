from torch.utils.data.dataset import Dataset
from load_data import *

# Some Class to specify our Dataset for the DataLoader
class MyDataset(Dataset):
	def __init__(self, filename):
		self.tok_text, self.sents = tokenize_text(filename)
		self.vocab = make_vocab(self.sents)
		self.labels = make_labels(sent_list)
		self.unlabeled_sents = make_unlabeled(sent_list)

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		return self.unlabeled_sents[idx], self.labels[idx]
	
	def get_vocab(self):
		return vocab