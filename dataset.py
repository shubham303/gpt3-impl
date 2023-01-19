
import torch
import torch.nn as nn
from torch.nn import functional as F


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, block_size, transform):
        self.transform = transform    
        self.text = text
        self.block_size = block_size # here are all the unique characters that occur in this text
        self.chars = sorted(list(set(text)))
      
    def __len__(self):
        return len(self.text) - self.block_size - 1

    def __getitem__(self, idx):
        return self.transform(self.text[idx:idx+self.block_size]) , self.transform(self.text[idx+1:idx+self.block_size+1])




def get_data_loaders(path, block_size, batch_size):
    
    # load text
    with open(path, 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # define transform function that takes string list of characters and returns a tensor of integers
    def transform(x):
        # convert list of characters to list of integers
        x = encode(x)
        return torch.tensor(x, dtype=torch.long)


    # divide text into train and val
    train_data = text[:int(len(text)*0.9)]
    val_data = text[int(len(text)*0.9):]

    train_dataset = TextDataset(train_data, block_size, transform)
    val_dataset = TextDataset(val_data, block_size, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader
