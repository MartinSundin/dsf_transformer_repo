import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from data import RandomLanguageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import math, os
import pandas as pd

"""
A minimal example of classifying an integer sequence as a series of tokens using the layers Embedding, PositionalEncoding, TransformerEncoder and a Dense network.
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class LanguageClassifier(nn.Module):
    def __init__(self,embedding_size=8, num_layers=1):
        super(LanguageClassifier, self).__init__()
        self.embedding = nn.Embedding(10, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=embedding_size)
        self.tf_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embedding_size,2)
        
    def forward(self, x):
        x = x.transpose(0,1)     # (batch_size, seq_len) -> (seq_len, batch_size)
        x = self.embedding(x)    # (seq_len, batch size) -> (seq_len, batch size, embed_dim)
        x = self.pos_encoding(x) # (seq_len, batch size, embed_dim) -> (seq_len, batch size, embed_dim)
        x = self.tf_encoder(x)   # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = x.mean(dim=0)        # (seq_len, batch_size, embed_dim) -> (batch_size, embed_dim)
        x = F.relu(x)            # (batch_size, embed_dim) -> (batch_size, embed_dim)
        x = self.linear(x)       # (batch_size, embed_dim) -> (batch_size, 2)
        return x
    

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 64
    num_layers = 10
    embedding_size = 32
    experiment_name = f'seq_class_{num_layers}_layer_{embedding_size}_emb_size2'
    os.makedirs(experiment_name, exist_ok=True)

    dataset = RandomLanguageDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = LanguageClassifier(embedding_size=embedding_size, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    loss_history = []
    acc_history = []
    acc_iters = []

    iter_num = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for i, (y_batch, X_batch) in tqdm(enumerate(dataloader), total=len(dataset) // batch_size + 1):
            optimizer.zero_grad()

            y_hat_batch = model(X_batch)
            y_batch_onehot = F.one_hot(y_batch.reshape(-1), num_classes=2).to(torch.float32)
            loss = criterion(y_hat_batch, y_batch_onehot)
            
            if iter_num % 10 == 0:
                acc = (y_hat_batch.argmax(dim=1) == y_batch_onehot.argmax(dim=1)).sum() / y_hat_batch.shape[0]
                acc_history.append(acc.item())
                acc_iters.append(iter_num)

            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()
            iter_num += 1
        print('Loss:', sum(loss_history[-25:])/25)
        print('Accuracy:', sum(acc_history[-10:])/10)
        pd.Series(loss_history).to_csv(os.path.join(experiment_name, 'loss.csv'))
        pd.Series(acc_history,index=acc_iters).to_csv(os.path.join(experiment_name, 'accuracy.csv'))

    # Plot historic loss
    plt.plot(loss_history)
    pd.Series(loss_history).rolling(25).mean().plot()
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.show()

    # Plot historic accuracy
    plt.plot(acc_iters, acc_history)
    pd.Series(acc_history, index=acc_iters).rolling(25).mean().plot()
    plt.title('Accuracy history')
    plt.xlabel('Iteration')
    plt.show()



    # TODO: Get self attention layer output
