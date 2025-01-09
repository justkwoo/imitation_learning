import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from mlp_constants import *

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class ExpertDataset(Dataset):
    def __init__(self, expert_filepath, transform=None, target_transform=None):
        self.expert_data = pd.read_pickle(expert_filepath)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.expert_data['state'])

    def __getitem__(self, idx):
        states = self.expert_data['state'][idx]
        actions = self.expert_data['action'][idx]
        if self.transform:
            states = self.transform(states)
        if self.target_transform:
            actions = self.target_transform(actions)
            
        interval_size = int(1080 / INPUT_LAYER_SIZE)

        return torch.tensor(states).view(INPUT_LAYER_SIZE, interval_size).mean(dim=1), torch.tensor(actions)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, input):
        return self.mlp(input)


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders
def train_model():
    batch_size = 64
    epochs = EPOCHS
    print(f"Using device: {DEVICE}")
    print(f"INPUT SIZE: {INPUT_LAYER_SIZE} | HIDDEN SIZE: {HIDDEN_LAYER_SIZE}")
    print(f"TRAIN DATASET: {TRAIN_DATASET}")
    
    # load dataset
    file_path = f'/Users/keonwookim/Desktop/CPEN_391/imitation_learning/imitation_pkg/expert_data/{TRAIN_DATASET}.pkl'
    dataset = ExpertDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # set up MLP
    path_to_model = f"/Users/keonwookim/Desktop/CPEN_391/imitation_learning/imitation_pkg/saved_models/{MODEL_TO_TRAIN}.pth"
    model = NeuralNetwork(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE).to(DEVICE)
    
    # uncomment it if need to train the existing model
    # model.load_state_dict(torch.load(path_to_model, map_location = DEVICE, weights_only=True))
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # the len of dataloader is dataset/batch_size
    #https://machinelearningmastery.com/creating-a-training-loop-for-pytorch-models/
    for epoch in range(epochs):
        epoch_loss = 0.0
      
        for states, actions in dataloader:
          # Move data to device
          states, actions = states.to(DEVICE), actions.to(DEVICE)
              
          # forward pass
          outputs = model(states)
          loss = criterion(outputs, actions)
          
          optimizer.zero_grad()
          
          # backward pass
          loss.backward()
          
          # update weights
          optimizer.step()
          
          # calculate loss
          epoch_loss += loss.item() * batch_size
        
        # epoch_loss:
        epoch_loss /= len(dataset)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.5f}")
    
    torch.save(model.to("cpu").state_dict(), path_to_model)
    print(f"Model training done and saved as {path_to_model}")
    

# run training
train_model()
    
    