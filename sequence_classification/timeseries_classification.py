import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from data import RandomLanguageDataset
from torch.utils.data import DataLoader
from sequence_classification import PositionalEncoding
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

"""
A minimal example of classifying an integer sequence as a timeseries using the layers 1DConvolutional, PositionalEncoding, TransformerEncoder and Dense.
"""
    

class LanguageClassifier(nn.Module):
    def __init__(self, embed_dim=8,num_layers=1):
        super(LanguageClassifier, self).__init__()
        self.conv_1d = nn.Conv1d(1, embed_dim, kernel_size=5)
        self.pos_encoding = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=embed_dim)
        self.tf_encoder = nn.TransformerEncoder(encoder_layer=decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_dim, 2)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1]) # (batch_size, seq_len) -> (batch_size, 1, seq_len)
        x = self.conv_1d(x)                      # (batch_size, 1, seq_len) -> (batch_size, embed_dim, seq_len')
        x = x.transpose(2,1).transpose(1,0)      # (batch_size, embed_dim, seq_len') -> (seq_len', batch_size, embed_dim)
        x = self.pos_encoding(x)                 # (seq_len', batch_size, embed_dim) -> (seq_len', batch_size, embed_dim)
        x = self.tf_encoder(x)                   # (seq_len', batch_size, embed_dim) -> (seq_len', batch_size, embed_dim)
        x = x.mean(dim=0)                        # (seq_len', batch_size, embed_dim) -> (batch_size, embed_dim)
        x = self.linear(x)                       # (batch_size, embed_dim) -> (batch_size, 2)
        return x
    

if __name__ == "__main__":
    num_epochs = 5
    batch_size = 64
    num_layers = 1
    embed_dim = 8
    experiment_name = f'times_class_{num_layers}_layer_embed_{embed_dim}'
    os.makedirs(experiment_name, exist_ok=True)

    dataset = RandomLanguageDataset(p=0.1)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = LanguageClassifier(embed_dim=embed_dim, num_layers=num_layers)
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

            y_hat_batch = model(X_batch.to(torch.float32))
            y_batch_onehot = F.one_hot(y_batch.reshape(-1), num_classes=2).to(torch.float32)
            loss = criterion(y_hat_batch, y_batch_onehot)
            
            if iter_num % 5 == 0:
                acc = (y_hat_batch.argmax(dim=1) == y_batch_onehot.argmax(dim=1)).sum() / y_hat_batch.shape[0]
                acc_history.append(acc.item())
                acc_iters.append(iter_num)

            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()
            iter_num += 1

        print('Loss:', loss_history[-1])
        print('Accuracy:', acc_history[-1])
        pd.Series(loss_history).to_csv(os.path.join(experiment_name, 'loss.csv'))
        pd.Series(acc_history,index=acc_iters).to_csv(os.path.join(experiment_name, 'accuracy.csv'))


    # Plot historic loss
    plt.plot(loss_history)
    pd.Series(loss_history).rolling(25).mean().plot()
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(acc_iters, acc_history)
    pd.Series(acc_history, index=acc_iters).rolling(25).mean().plot()
    plt.title('Accuracy history')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()

    # # Visualise CNN filters
    # weights = model.conv_1d._parameters['weight'].detach().numpy()
    # fig, ax = plt.subplots(weights.shape[0],1)
    # for k in range(weights.shape[0]):
    #     ax[k].plot(weights[k,0,:]) # , 0.9/weights.shape[0])
    #     ax[k].plot(weights[k,0,:]*0.0, color='red')
    #     ax[k].set_title(f"Filter {k}")
    # plt.show()

    foo = -1
