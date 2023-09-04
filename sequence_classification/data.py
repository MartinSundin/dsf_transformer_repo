import numpy as np
import torch
from torch.utils.data import Dataset

"""
UP and DOWN sequence dataset
"""

class RandomLanguageDataset(Dataset):
    """
    Dataset that generates a random UP or DOWN sequence
    """
    def __init__(self, seq_len=100, num_seqs=10_000, probs=[1/2,1/2], p=0.6):
        self.seq_len = seq_len
        self.num_seqs = num_seqs
        self.lang_probs = np.array(probs)
        self.probs_up = np.array([0.2*p, 1.0-p, 0.8*p])
        self.probs_down = np.array([0.8*p, 1.0-p, 0.2*p])

    def __len__(self):
        return self.num_seqs
    
    def __getitem__(self, i):
        lang_label = np.argmax(self.lang_probs.cumsum() >= np.random.rand())
        match lang_label:
            case 0: move_probs = self.probs_down.cumsum()
            case 1: move_probs = self.probs_up.cumsum()
        moves = np.argmax(np.tile(move_probs.reshape(-1,1),self.seq_len) >= np.tile(np.random.rand(self.seq_len).reshape(-1,1),3).transpose(), axis=0)
        sentence = np.cumsum(moves - 1) % 10
        return torch.Tensor([lang_label]).to(torch.long), torch.Tensor(sentence).to(torch.int)

if __name__ == "__main__":
    dataset = RandomLanguageDataset(seq_len=35)
    for i in range(10):
        label, label_seq = dataset[i]
        print(i, label, label_seq)
