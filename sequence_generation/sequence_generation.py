import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from data import RandomShakespeareSentence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import os
from datetime import datetime

"""
A minimal example of generating a sequence of characters using the layers Embedding, PositionalEncoding, TransformerDecoder and Dense.
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
    

class SentenceGenerator(nn.Module):
    def __init__(self, num_chars, max_length, embed_dim=8, num_layers=1):
        super(SentenceGenerator, self).__init__()
        self.num_chars = num_chars
        self.embedding = nn.Embedding(num_chars+3, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.memory = torch.from_numpy(np.zeros(1).astype(np.float32))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=embed_dim)
        self.tf_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_dim, num_chars+3)

    def forward(self, x):
        if x.shape[0] > 1:
            padding_mask = (x == 35)
            padding_mask = torch.concat([torch.zeros((padding_mask.shape[0],1)).to(torch.bool), padding_mask[:,:-1]], dim=1)
        else:
            padding_mask = None

        x = x.transpose(0,1)        # (batch_size, seq_len) -> (seq_len, batch_size)
        x = self.embedding(x)       # (seq_len, batch_size) -> (seq_len, batch_size, embed_dim)
        x = self.pos_encoding(x)    # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, embed_dim)

        mask = torch.triu(torch.full((x.shape[0], x.shape[0]), float('-inf')), diagonal=1) 
                                    # (seq_len, seq_len)
        memory = torch.tile(self.memory, x.shape) 
                                    # (seq_len, batch_size, embed_dim)

        x = self.tf_decoder.forward(tgt=x, memory=memory, tgt_mask=mask, tgt_key_padding_mask=padding_mask)
                                    # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = self.linear(x)          # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, num_chars+3)
        return x.transpose(0,1)     # (seq_len, batch_size, num_chars+3) -> (batch_size, seq_len, num_chars+3)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    num_epochs = 10
    batch_size = 64
    num_layers = 1
    embed_dim = 8

    dataset = RandomShakespeareSentence()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = SentenceGenerator(num_chars=len(dataset.characters), 
                              max_length=dataset.max_length+3, 
                              embed_dim=embed_dim, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)


    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, cooldown=100, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=0.1, step_size_up=2000, cycle_momentum=False)
    loss_history = []
    acc_history = []
    acc_iters = []

    iter_num = 0 if len(loss_history) == 0 else len(loss_history)
    for epoch in range(num_epochs):
        for i, (X_batch, X_target) in tqdm(enumerate(dataloader), total=len(dataset) // batch_size + 1):
            optimizer.zero_grad()
            y_hat_batch = model(X_batch)
            y_batch_onehot = F.one_hot(X_target.to(torch.long), num_classes=dataset.num_characters+3).to(torch.float32)
            loss = criterion(y_hat_batch.transpose(1,2), y_batch_onehot.transpose(1,2))
            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()

            
            # scheduler.step(loss)
            scheduler.step()

            if iter_num % 10 == 0:
                acc = ((y_hat_batch.argmax(dim=2) == X_target)*(X_target != 35)).sum() / (X_target != 35).sum() # (y_hat_batch.shape[0]*y_hat_batch.shape[1])
                # acc = (F.softmax(y_hat_batch,dim=2) * y_batch_onehot).sum(dim=2).mean()
                acc_history.append(acc.item())
                acc_iters.append(iter_num)

            iter_num += 1


    # Plot historic loss
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.show()

    plt.plot(acc_iters, acc_history)
    plt.title('Accuracy history')
    plt.xlabel('Iteration')
    plt.show()


    foo = -1 # debugging breakpoint


    # # TODO: Get self attention layer output?
    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook
    # model.tf_encoder.layers[0].register_forward_hook(get_activation('encoder_penultimate_layer'))
