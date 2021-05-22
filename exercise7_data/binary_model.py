
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np


class BinaryNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 1)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 1)
        #self.s = torch.nn.Softmax(dim=1)
        self.s = torch.nn.Sigmoid()

    def forward(self, x):
        #x = F.relu( self.fc1(x) )
        x = self.fc1(x)
        #x = F.relu( self.fc2(x) )
        x = self.s( self.fc2(x) )
        return x


def train_binary_model(model, device, train_loader, test_loader, epochs, silent):

  optimizer = optim.Adam(model.parameters())

  for epoch in tqdm(range(1, epochs+1), desc="Epoch", unit="epoch"):

      # Perform training step\n",
      train_acc, train_loss = _train(model, device, train_loader, optimizer)
        
      # Perform test step\n",
      test_acc, test_loss  = _test(model, device, test_loader)
    
      # Print results\n",
      if not silent:
        print(f"Epoch: {epoch}, Avg Train Loss: {train_loss:.4f}, Train Acc: {(100. * train_acc):.2f}%")
        print(f"Epoch: {epoch}, Avg Test Loss: {test_loss:.4f}, Test Acc: {(100. * test_acc):.2f}%\n")
    
  return (train_acc, test_acc), (train_loss, test_loss)
        

def _train(model, device, train_loader, optimizer):
  
  model.train()
  criterion = nn.BCEWithLogitsLoss() #nn.BCELoss()  #nn.BCEWithLogitsLoss() ##

  losses, top1_acc = list(), list()

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
        
      losses.append(loss.item())
      
      preds = y_hat.detach().cpu().squeeze(1).numpy()
      labels = y.detach().cpu().numpy()
      acc = accuracy(preds.round(), labels)
      top1_acc.append(acc)
        
  #print(top1_acc)
    
  top1_avg = np.mean(top1_acc)
  test_loss = np.mean(losses)

  return top1_avg, test_loss


def _test(model, device, test_loader):
  model.eval()
  criterion = nn.BCEWithLogitsLoss()

  losses, top1_acc = list(), list()

  with torch.no_grad():
    for data, target in test_loader:
      x, y = data.to(device), target.to(device)
      y_hat = model(x)
      test_loss = criterion(y_hat, y.unsqueeze(1).float())
    
      losses.append(test_loss.item())
      
      preds = y_hat.detach().cpu().squeeze(1).numpy()
      labels = y.detach().cpu().numpy()
      acc = accuracy(preds.round(), labels)
      top1_acc.append(acc)
  
  top1_avg = np.mean(top1_acc)
  test_loss = np.mean(losses)
    
  return top1_avg, test_loss
      
    
def accuracy(preds, labels):
  return (preds == labels).mean()