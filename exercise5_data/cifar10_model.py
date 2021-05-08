
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np



class Cifar10Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (5, 5))
        self.conv2 = nn.Conv2d(32, 32, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.pool( F.relu( self.conv1(x) ) )
        x = self.pool( F.relu( self.conv2(x) ) )
        x = torch.flatten(x, 1) 
        x = torch.tanh( self.fc1(x) )
        x = self.fc2(x)
        return x


def train_cifar10_model(model, device, train_loader, test_loader, epochs):

  optimizer = optim.Adam(model.parameters())

  for epoch in tqdm(range(1, epochs+1), desc="Epoch", unit="epoch"):
      # Perform training step\n",
      train_acc, train_loss  = train(model, device, train_loader, optimizer)
      
      # Perform test step\n",
      test_acc, test_loss  = test(model, device, test_loader)
    
      # Print results\n",
      print(f"Epoch: {epoch}, Avg Train Loss: {train_loss:.4f}, Train Acc: {(100. * train_acc):.2f}%")
      print(f"Epoch: {epoch}, Avg Test Loss: {test_loss:.4f}, Test Acc: {(100. * test_acc):.2f}%\n")
        

def train(model, device, train_loader, optimizer):
  
  model.train()
  criterion = nn.CrossEntropyLoss()

  losses, top1_acc = list(), list()

  for batch_idx, (data, target) in enumerate(train_loader):
      # get the inputs; data is a list of [inputs, labels]
      x, y = data.to(device), target.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      y_hat = model(x)
      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
      preds = np.argmax(y_hat.detach().cpu().numpy(), axis=1)
      labels = y.detach().cpu().numpy()
      acc = accuracy(preds, labels)
      top1_acc.append(acc)
      losses.append(loss.item())

  top1_avg = torch.mean(torch.tensor(top1_acc)).detach().cpu().numpy()
  avg_loss = torch.mean(torch.tensor(losses)).detach().cpu().numpy()
  
  return top1_avg, avg_loss



def test(model, device, test_loader):
  model.eval()
  criterion = nn.CrossEntropyLoss()

  losses, top1_acc = list(), list()

  with torch.no_grad():
    for data, target in test_loader:
      x, y = data.to(device), target.to(device)
      y_hat = model(x)
      test_loss = criterion(y_hat, y)
      preds = np.argmax(y_hat.detach().cpu().numpy(), axis=1)
      labels = y.detach().cpu().numpy()
      acc1 = accuracy(preds, labels)
      
      losses.append(test_loss.item())
      top1_acc.append(acc1)
  
  top1_avg = np.mean(top1_acc)
  test_loss = np.mean(losses)
    
  return top1_avg, test_loss


def accuracy(preds, labels):
  return (preds == labels).mean()