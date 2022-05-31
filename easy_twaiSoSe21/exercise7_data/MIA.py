import torch
import numpy as np

from binary_model import BinaryNet, train_binary_model


def getTopk(data, top):
    """
    Returns the top k maximum entries of the given data.
    """
    
    return np.array([sorted(s, reverse=True)[:top] for s in data])


def predict_(model, device, data_loader):
    """
    Iterates the given data_loader and collects the predicition vectors
    of the given model in a list.
    Note: It is essential for this attack to apply the softmax function 
          DIRECTLY AFTER receiving the prediction vector
    """

    softmax = torch.nn.Softmax(dim=1)

    preds = list()
    with torch.no_grad():
        for data, _ in data_loader:
            x = data.to(device)
            preds += softmax( model(x) ).tolist()
    return preds


def get_mia_accuracy(model, device, train_loader, test_loader, train_kwargs, test_kwargs, size, test_size:int = 1000):

  model.eval()

  # IN
  preds_IN = predict_( model, device, train_loader )
  labels_IN = np.ones(size)

  # OUT
  preds_OUT = predict_( model, device, test_loader )
  labels_OUT = np.zeros(size)

  # ALL = IN + OUT
  preds = np.concatenate([preds_IN, preds_OUT])
  labels = np.concatenate([labels_IN, labels_OUT])

  # Top 3 prediction values for every data point
  preds = getTopk(preds, 3)

  # Create attack dataset -> train (and test)
  attack_dataset = torch.utils.data.TensorDataset(torch.Tensor(preds[test_size:-test_size]).type(torch.FloatTensor), torch.Tensor(labels[test_size:-test_size]).type(torch.LongTensor) )
  train_attacker_loader = torch.utils.data.DataLoader(attack_dataset, **train_kwargs)

  attack_dataset_test = torch.utils.data.TensorDataset(torch.Tensor(np.concatenate([preds[:test_size],preds[-test_size:]])).type(torch.FloatTensor), torch.Tensor(np.concatenate([labels[:test_size],labels[-test_size:]])).type(torch.LongTensor) )
  test_attacker_loader = torch.utils.data.DataLoader(attack_dataset_test, **test_kwargs)

  attack_model = BinaryNet().to(device)
  (acc, test_acc), (loss, test_loss) = train_binary_model(attack_model, device, train_attacker_loader, test_attacker_loader, 10, silent=True)

  print("### MIA RESULTS ###")
  print("Accuracy")
  print("Train:", acc)
  print("Test:", test_acc)
  print("-"*20)
  print("Loss")
  print("Train:", loss)
  print("Test:", test_loss)