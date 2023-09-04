import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from data import MNISTAnnotatedDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import os
from PIL import Image
import yaml


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
    

class ImageAnnotator(nn.Module):
    def __init__(self, num_chars, end_token, embed_dim=8, num_layers=2, num_decoders=2, num_linear=2):
        super(ImageAnnotator, self).__init__()
        # Image network
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=embed_dim, kernel_size=(3,3), stride=2)
        self.cnn2 = nn.Conv2d(in_channels=embed_dim,out_channels=embed_dim, kernel_size=(4,4), stride=2)
        self.cnn_linear1 = nn.Linear(25, 20)
        self.cnn_linear2 = nn.Linear(20, 16)

        # Text network
        self.end_token = end_token
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=15+2, embedding_dim=embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=16)
        self.decoders = []
        for _ in range(num_decoders):
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=embed_dim)
            tf_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
            self.decoders.append(tf_decoder)
        decoder_layer2 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=embed_dim)
        self.tf_decoder2 = nn.TransformerDecoder(decoder_layer=decoder_layer2, num_layers=num_layers)

        # Dense network
        linear_out_layers = []
        for i in range(num_linear):
            in_size = embed_dim if i == 0 else num_chars
            linear_out = nn.Linear(in_size, num_chars)
            linear_out_layers.append(linear_out)
            if i < num_linear - 1:
                linear_out_layers.append(nn.ReLU)

        self.linear = nn.Sequential(*linear_out)


        # Mask
        self.mask = torch.triu(torch.full((13, 13), float('-inf')), diagonal=1)

    def forward(self, X, w, use_mask=True):
        # Image network
        X = X.reshape((X.shape[0],1) + X.shape[1:]) # (batch_size,im_size,im_size) -> (batch_size,1,im_size,im_size)
        X = self.cnn1(X) # (batch_size,1,im_size,im_size) -> (batch_size,embed_dim,im_size',im_size')
        X = F.relu(X) # (batch_size,embed_dim,im_size',im_size') -> (batch_size,embed_dim,im_size',im_size')
        X = self.cnn2(X) # (batch_size,embed_dim,im_size',im_size') -> (batch_size,embed_dim,im_size'',im_size'')
        X = F.relu(X) # (batch_size,embed_dim,im_size'',im_size'') -> (batch_size,embed_dim,im_size'',im_size'')
        memory = torch.flatten(X, start_dim=2)
        memory = self.cnn_linear1(memory)
        memory = F.relu(memory)
        memory = self.cnn_linear2(memory)
        memory = memory.transpose(1,2).transpose(0,1) # (batch_size,embed_dim,im_size''*im_size'') -> (im_size''*im_size'',batch_size,embed_dim)
        
        # Text network
        w = self.embedding(w) # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        w = w.transpose(0,1) # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        w = self.pos_encoding(w) # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, embed_dim)
        mask = self.mask if use_mask else None
        # padding_mask = self.padding_mask if use_mask else None
        padding_mask = None
        for decoder in self.decoders:
            w = decoder(tgt=w, memory=memory, tgt_mask=mask, tgt_key_padding_mask=padding_mask)
        # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, embed_dim)
        # w = self.decoder2(tgt=w, memory=memory, tgt_mask=mask, tgt_key_padding_mask=padding_mask)
        # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, embed_dim)

        # Dense network
        x = self.linear(w)     # (seq_len, batch_size, embed_dim) -> (seq_len, batch_size, num_chars)
        return x.transpose(0,1) # (seq_len, batch_size, num_chars) -> (batch_size, seq_len, num_chars)
    
if __name__ == "__main__":
    # Hardcoded training parameters
    num_layers = 2
    num_decoders = 5
    embed_dim = 8
    num_linear = 4

    # yaml file structure explained in data.py
    with open('./image_classification/config.yaml','r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        example_folder = config['example_folder']

    dataset = MNISTAnnotatedDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ImageAnnotator(dataset.end_token+1, dataset.end_token,
                           num_decoders=num_decoders, num_layers=num_layers,
                           embed_dim=embed_dim,num_linear=num_linear)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3,max_lr=1e-2,step_size_up=1000,step_size_down=1000)

    loss_history = []
    acc_iters = []
    acc_history = []
    iter_num = -1
    for epoch in range(100):
        print(f'Epoch {epoch+1}')
        for y, X in tqdm(dataloader, total=len(dataloader)):
            iter_num += 1
            optimizer.zero_grad()

            y_out = y[:,1:]
            y_hat = model(X, y[:,:-1])
            y_out = F.one_hot(y_out,num_classes=dataset.end_token+1).to(torch.float32)

            loss = criterion(y_hat, y_out)
            loss_history.append(loss.item())
            if iter_num % 10 == 0:
                acc_iters.append(iter_num)
                current_acc = (y_hat.argmax(dim=2) == y_out.argmax(dim=2)).sum() / (y_out.shape[0]*y_out.shape[1])
                acc_history.append(current_acc.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f'Loss = {sum(loss_history[-10:])/10}')
        print(f'Accuracy = {sum(acc_history[-10:])/10}')

        for j,i in enumerate([21,23,16,27,20,35,18,15,17,19]):
            y, X = dataset[i]
            X = X.reshape((1,) + X.shape)
            w_pred = [dataset.start_token]
            last_char = -1
            while last_char != dataset.end_token and len(w_pred) < 16:
                w_pred_tensor = torch.tensor(w_pred).reshape(1,-1)
                with torch.no_grad():
                    char_preds = model(X, w_pred_tensor, use_mask=False)
                    last_char = char_preds[0,-1,:].argmax().item()
                w_pred.append(last_char)

            print(f'{j}: {"".join([dataset.letters[c] for c in w_pred if c not in [dataset.start_token, dataset.end_token]])}')


        for name in os.listdir(example_folder):
            filename = os.path.join(example_folder, name)
            image = Image.open(filename)
            image = 1.0 - np.array(image).mean(axis=2) / 255
            X = torch.from_numpy(image).to(torch.float32).reshape(1,28,28)
            w_pred = [dataset.start_token]
            last_char = -1
            while last_char != dataset.end_token and len(w_pred) < 16:
                w_pred_tensor = torch.tensor(w_pred).reshape(1,-1)
                with torch.no_grad():
                    char_preds = model(X, w_pred_tensor, use_mask=False)
                    last_char = char_preds[0,-1,:].argmax().item()
                w_pred.append(last_char)

            print(f'{name}: {"".join([dataset.letters[c] for c in w_pred if c not in [dataset.start_token, dataset.end_token]])}')


    # Plot loss history
    plt.plot(loss_history)
    pd.Series(loss_history).rolling(10).mean().plot()
    plt.title('Loss History')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()

    foo = -1 # debugging checkpoint

     # Plot accuracy history
    plt.plot(acc_iters, acc_history)
    pd.Series(acc_history,index=acc_iters).rolling(10).mean().plot()
    plt.title('Accuracy History')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.show()

    foo = -1 # debugging checkpoint

    # Plot a few examples
    for i in [21,23,16,27,20,35,18,15,17,19]:
        y, X = dataset[i]
        X = X.reshape((1,) + X.shape)
        w_pred = [dataset.start_token]
        last_char = -1
        while last_char != dataset.end_token and len(w_pred) < 16:
            w_pred_tensor = torch.tensor(w_pred).reshape(1,-1)
            with torch.no_grad():
                char_preds = model(X, w_pred_tensor, use_mask=False)
                last_char = char_preds[0,-1,:].argmax().item()
            w_pred.append(last_char)

        plt.imshow(X[0,:,:].detach().numpy(), cmap='gray')
        plt.title(f'({i}), {dataset.decode(y)}. Predicted: {dataset.decode(torch.tensor(w_pred))}')
        plt.axis('off')
        plt.show()

        foo = -1 # debugging checkpoint

    for name in os.listdir(example_folder):
        filename = os.path.join(example_folder, name)
        image = Image.open(filename)
        image = 1.0 - np.array(image).mean(axis=2) / 255
        X = torch.from_numpy(image).to(torch.float32).reshape(1,28,28)
        w_pred = [dataset.start_token]
        last_char = -1
        while last_char != dataset.end_token and len(w_pred) < 16:
            w_pred_tensor = torch.tensor(w_pred).reshape(1,-1)
            with torch.no_grad():
                char_preds = model(X, w_pred_tensor, use_mask=False)
                last_char = char_preds[0,-1,:].argmax().item()
            w_pred.append(last_char)

        plt.imshow(X[0,:,:].detach().numpy(), cmap='gray')
        plt.title(f'example, {name}. Predicted: {dataset.decode(torch.tensor(w_pred))}')
        plt.axis('off')
        plt.show()
        
        foo = -1 # debugging checkpoint


    foo = -1 # debugging checkpoint
