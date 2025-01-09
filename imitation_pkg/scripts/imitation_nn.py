#!/usr/bin/env python3

import numpy as np
import torch 
from torch import nn, optim 

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, device):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, output_size), 
        )    
        
        # define device as global
        self.device = device
        
        # define model to mlp
        model = self.mlp
        
        # attach to gpu
        model.to(device)
        
        # set up optimizer
        self.optimizer = optim.Adam(model.parameters(), learning_rate)
        
        # set up loss function for training
        self.loss_func = nn.MSELoss()
    
    # forward pass func
    def forward(self, input):
        return self.mlp(input)
    
    # func to get predicted outputs
    def predict_output(self, scan_msg): 
        output = self(scan_msg)
        
        return output.detach()
    
    # func to compute loss compared to expert data
    def compute_loss(self, scan_msg, expert_angle, expert_speed):
        expert_output = torch.tensor([expert_angle, expert_speed], device = self.device, dtype = torch.float32)
        scan_tensor = scan_msg.to(self.device)
        
        # call predcit output to get outputs
        mlp_output = self.predict_output(scan_tensor)
        
        # compute loss 
        loss = self.loss_func(mlp_output, expert_output)
        
        return loss.item()
    
