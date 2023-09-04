import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import yaml

"""
config.yaml has the fields
example_folder: # path to folder with 28 x 28 grayscale example images
mnist_folder: # folder to cache the mnist training data in
"""

with open('image_classification/config.yaml','r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    mnist_root_dir = config['mnist_folder']

class MNISTAnnotatedDataset(Dataset):
    def __init__(self):
        self.mnist = MNIST(root=mnist_root_dir, download=True)
        self.name_dict = [
            'zero',
            'one',
            'two',
            'three',
            'four',
            'five',
            'six',
            'seven',
            'eight',
            'nine'
        ]
        self.letters = list(sorted(set([c for w in self.name_dict for c in w])))
        self.max_len = 12 #  max(map(len, self.name_dict))
        self.start_token = len(self.letters)
        self.end_token = len(self.letters) + 1

    def __len__(self):
        return len(self.mnist)
    
    def decode(self, y):
        return ''.join(self.letters[c] for c in y.detach().numpy() if c not in [self.start_token, self.end_token])
    
    def __getitem__(self, i):
        X_image, num_label = self.mnist[i]
        encoded_label = [self.letters.index(c) for c in self.name_dict[num_label]]
        label_tensor = torch.tensor([self.start_token] \
                                    + encoded_label \
                                    + [self.end_token]*(self.max_len - len(encoded_label) + 1)).to(torch.long)
        X_tensor = torch.from_numpy((np.array(X_image) / 255).astype(np.float32))
        return label_tensor, X_tensor

if __name__ == "__main__":
    dataset = MNISTAnnotatedDataset()
    y, X = dataset[175]
    label = dataset.decode(y)
    
    import matplotlib.pyplot as plt

    plt.imshow(X.detach().numpy())
    plt.title(label)
    plt.show()
