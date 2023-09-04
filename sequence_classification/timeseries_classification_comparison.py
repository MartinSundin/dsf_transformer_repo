import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from data import RandomLanguageDataset
from torch.utils.data import DataLoader
from timeseries_classification import LanguageClassifier as TransformerClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

"""
Comparison of the Transformer model with a simple CNN model.
"""
    
class CNNClassifier(nn.Module):
    def __init__(self, embed_dim=8):
        super(CNNClassifier, self).__init__()
        self.conv_1d = nn.Conv1d(1,embed_dim, kernel_size=5)
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.conv_1d(x)
        x = F.relu(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    num_epochs = 25
    batch_size = 64
    num_layers = 1
    embed_dim = 8
    experiment_name = f'times_class_{num_layers}_layer_embed_{embed_dim}'
    os.makedirs(experiment_name, exist_ok=True)

    dataset = RandomLanguageDataset(p=0.1)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    transformer_model = TransformerClassifier(embed_dim=embed_dim, num_layers=num_layers)
    cnn_model = CNNClassifier(embed_dim=embed_dim)
    criterion = nn.CrossEntropyLoss()
    transformer_optimizer = optim.SGD(transformer_model.parameters(), lr=1e-2)
    cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2)
    
    loss_transformer = []
    acc_transformer = []
    loss_cnn = []
    acc_cnn = []
    acc_iters = []

    iter_num = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for i, (y_batch, X_batch) in tqdm(enumerate(dataloader), total=len(dataset) // batch_size + 1):
            # Transformer model
            transformer_optimizer.zero_grad()

            y_hat_batch = transformer_model(X_batch.to(torch.float32))
            y_batch_onehot = F.one_hot(y_batch.reshape(-1), num_classes=2).to(torch.float32)
            loss = criterion(y_hat_batch, y_batch_onehot)
            
            if iter_num % 5 == 0:
                acc = (y_hat_batch.argmax(dim=1) == y_batch_onehot.argmax(dim=1)).sum() / y_hat_batch.shape[0]
                acc_transformer.append(acc.item())
                acc_iters.append(iter_num)

            loss_transformer.append(loss.item())

            loss.backward()
            transformer_optimizer.step()

            # CNN Model
            transformer_optimizer.zero_grad()

            y_hat_batch = cnn_model(X_batch.to(torch.float32))
            y_batch_onehot = F.one_hot(y_batch.reshape(-1), num_classes=2).to(torch.float32)
            loss = criterion(y_hat_batch, y_batch_onehot)
            
            if iter_num % 5 == 0:
                acc = (y_hat_batch.argmax(dim=1) == y_batch_onehot.argmax(dim=1)).sum() / y_hat_batch.shape[0]
                acc_cnn.append(acc.item())

            loss_cnn.append(loss.item())

            loss.backward()
            cnn_optimizer.step()


            iter_num += 1

        print('Transformer Loss:', loss_transformer[-1])
        print('Transformer Accuracy:', acc_transformer[-1])
        print('CNN Loss:', loss_cnn[-1])
        print('CNN Accuracy:', acc_cnn[-1])
        # pd.Series(loss_history).to_csv(os.path.join(experiment_name, 'loss.csv'))
        # pd.Series(acc_history,index=acc_iters).to_csv(os.path.join(experiment_name, 'accuracy.csv'))


    # Plot historic loss
    plt.plot(loss_transformer, label='transformer')
    pd.Series(loss_transformer).rolling(25).mean().plot(label='transformer')
    plt.plot(loss_cnn, label='cnn')
    pd.Series(loss_cnn).rolling(25).mean().plot(label='cnn')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    plt.plot(acc_iters, acc_transformer, label='Transformer')
    pd.Series(acc_transformer, index=acc_iters).rolling(25).mean().plot(label='transformer')
    plt.plot(acc_iters, acc_cnn, label='CNN')
    pd.Series(acc_cnn, index=acc_iters).rolling(25).mean().plot(label='cnn')
    plt.title('Accuracy history')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    foo = -1 # debugging breakpoint
