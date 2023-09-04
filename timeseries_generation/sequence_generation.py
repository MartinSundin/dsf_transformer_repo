import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from data import WeatherDataset
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
    

class WeatherPredictor(nn.Module):
    def __init__(self, embed_dim=2, num_layers=1, layer_size=8):
        super(WeatherPredictor, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=3,out_channels=embed_dim, kernel_size=8)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=25)
        self.memory = torch.from_numpy(np.ones(1).astype(np.float32))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=embed_dim)
        self.tf_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.linear1 = nn.Linear(embed_dim, layer_size)
        self.linear2 = nn.Linear(layer_size+13, layer_size)
        self.linear3 = nn.Linear(layer_size, 2)

    def forward(self, x, t):
        x = x.transpose(1,2)                # (batch_size, seq_len, 3) -> (batch_size, 3, seq_len)
        x = self.conv1d(x)                  # (batch_size, 3, seq_len) -> (batch_size, embed_dim, seq_len')
        x = x.transpose(2,1).transpose(1,0) # (batch_size, embed_dim, seq_len') -> (seq_len', batch_size, embed_dim)
        x = self.pos_encoding(x)            # (seq_len', batch_size, embed_dim) -> (seq_len', batch_size, embed_dim)
        mask = torch.triu(torch.full((x.shape[0], x.shape[0]), float('-inf')), diagonal=1)     # (seq_len', seq_len')
        memory = torch.tile(self.memory, x.shape)                                 # (seq_len', batch_size, embed_dim)
        x = self.tf_decoder.forward(tgt=x, memory=memory, tgt_mask=mask) 
                                            # (seq_len', batch_size, embed_dim) -> (seq_len', batch_size, embed_dim)
        x = self.linear1(x)                 # (seq_len', batch_size, embed_dim) -> (seq_len', batch_size, 8)
        x = F.relu(x)
        t = t[:, -x.shape[0]:, 0].transpose(0,1) # (batch_size, seq_len, 1) -> (seq_len', batch_size)
        t = F.one_hot(t.to(torch.long), num_classes=13)
                                            # (seq_len', batch_size, 1) -> (seq_len', batch_size, 13)
        x = torch.concat([x, t],dim=2)      # (seq_len', batch_size, 21)
        x = self.linear2(x)                 # (seq_len', batch_size, 8+13) -> (seq_len', batch_size, 8)
        x = F.relu(x)
        x = self.linear3(x)                 # (seq_len', batch_size, 8) -> (seq_len', batch_size, 2)
        return x.transpose(0,1)             # (seq_len', batch_size, 2) -> (batch_size, seq_len' ,2)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    # hardcoded parameters
    num_epochs = 25
    batch_size = 64
    num_layers = 1
    embed_dim = 8
    layer_size = 8

    dataset = WeatherDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = WeatherPredictor(embed_dim=embed_dim, num_layers=num_layers, layer_size=layer_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_history = []

    iter_num = 0 if len(loss_history) == 0 else len(loss_history)
    for epoch in range(num_epochs):
        for i, X_batch in tqdm(enumerate(dataloader), total=len(dataset) // batch_size + 1):
            optimizer.zero_grad()

            X_input = X_batch[:,:24,:]
            X_output = X_batch[:,3:,1:]
            t_input = X_batch[:,3:,:1]
            X_hat_batch = model(X_input, t_input)
            X_output = X_output[:,-X_hat_batch.shape[1]:,:]
            loss = criterion(X_hat_batch, X_output)

            loss.backward()
            optimizer.step()


            loss_history.append(loss.item())
            iter_num += 1


    # Plot historic loss
    std_loss = dataset.df.groupby('Month')[['tmax','tmin']].var().mean().mean()
    plt.plot(loss_history, label='Model Loss')
    plt.plot(np.arange(len(loss_history)),std_loss * np.ones(len(loss_history)), linestyle='dashed', label='Benchmark')
    plt.legend()
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.show()

    # Predict this years weather, year=2023
    with torch.no_grad():
        X_batch = dataset[767-14]
        for _ in range(12):
            X_input = X_batch.reshape((1,)+X_batch.shape)[:,-27-12:-3-12,:]
            t_input = ((X_input[:,:,:1] + 3) % 12) + 1
            X_hat = model(X_input, t_input)
            X_batch = torch.concat([
                X_batch, 
                torch.concat([
                    t_input[0,-1:,:],
                    X_hat[0,-1:,:]
            ], dim=1)
                    ],dim=0)

    df_prediction = pd.DataFrame(data=X_batch.detach().numpy())
    df_truth = dataset.df.iloc[-39:]
    for col in df_truth:
        df_prediction[col] = df_truth[col].values

    df_prediction[1].plot(label='predicted tmax', linestyle='dashed')
    df_prediction[2].plot(label='predicted tmin', linestyle='dashed')
    df_prediction['tmax'].plot(label='tmax')
    df_prediction['tmin'].plot(label='tmin')
    plt.legend()
    plt.title('Prediction and real temperature')
    plt.ylabel('Temperature')
    plt.xlabel('Timestep')
    plt.show()

    foo = -1 # debugging breakpoint