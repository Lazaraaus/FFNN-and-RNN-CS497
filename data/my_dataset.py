from torch.utils.data.dataset import Dataset
from data.load_data import *
import pdb

# Some Class to specify our Dataset for the DataLoader
class MyDataset(Dataset):
        def __init__(self, filename):
            self.sequences, self.vocab, self.embeddings = tokenize_text(filename)

            self.vocab2embedding = self.get_embedding_dict()
            self.vocab2index = self.get_vocab_index_dict()
            self.labels = make_labels(self.sequences)
            self.unlabeled_seqs = make_unlabeled(self.sequences)

        def __len__(self):
            return len(self.labels)
    
        def __getitem__(self, idx):
            #seq_as_embeddings = np.zeros((5, 100))

            return self.unlabeled_seqs[idx], self.labels[idx]  #seq_as_embeddings, self.vocab2embedding[self.labels[idx]]
        
        def get_vocab(self):
            return self.vocab

        def get_embeddings(self):
            return self.embeddings

        def get_embedding_dict(self):
            vocab2embedding = dict()

            for idx, token in enumerate(self.vocab):
                 vocab2embedding[token] = self.embeddings[idx]

            return vocab2embedding
        
        def get_vocab_index_dict(self):
            vocab2index = dict()
            for idx, word in enumerate(self.vocab):
                vocab2index[word] = idx

            return vocab2index

if __name__ == "__main__":
    test_dataset = MyDataset("test")
    #print(test_dataset.sequences[0:5])
    #print(2)
