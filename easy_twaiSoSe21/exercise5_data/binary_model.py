
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np


class BinaryNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = torch.sigmoid( self.fc3(x) )
        return x


def train_binary_model(model, device, train_loader, epochs):

  optimizer = optim.Adam(model.parameters())

  for epoch in tqdm(range(1, epochs+1), desc="Epoch", unit="epoch"):

      # Perform training step\n",
      _train(model, device, train_loader, optimizer)
        

def _train(model, device, train_loader, optimizer):
  
  model.train()
  criterion = nn.BCELoss()

  for batch_idx, (data, target) in enumerate(train_loader):
      # get the inputs; data is a list of [inputs, labels]
      x, y = data.to(device), target.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      y_hat = model(x)
      loss = criterion(y_hat, y.unsqueeze(1).float())
      loss.backward()
      optimizer.step()