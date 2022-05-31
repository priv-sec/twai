import torch
import numpy as np
from tqdm import trange



class Classifier(torch.nn.Module):

    """
    [0.75, 0.3, 0.2] -> 1 or 0 -> Training or testing point
    """

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.s = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.s( self.fc2(x) )
        return x


def train_classifier(model, device, train_loader, epochs):

    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in trange(1, epochs+1):

      for batch_idx, (data, target) in enumerate(train_loader):

        x, y = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1).float())
        loss.backward()
        optimizer.step()


def getTopk(data, top):

    sorted_data, _ = torch.sort(data, dim=1, descending=True)

    return torch.index_select(sorted_data, 1, torch.arange(0, top).to(data.device))


def predict_(model, device, data_loader):

    preds = list()
    with torch.no_grad():
        for data, _ in data_loader:
            x = data.to(device)
            preds += torch.nn.functional.softmax( model(x), dim=1 ).tolist()
    return preds


def get_mia_accuracy(model_class,
                     target_model,
                     target_epochs,
                     device,
                     target_loaders,
                     train_kwargs,
                     train,
                     shadow_dataset,
                     classifier_epochs,
                     load_shadow=False):

    target_model.eval()

    # Train Shadow Model
    shadow_model = model_class().to(device)

    if not load_shadow:
        train_shadow_dataset, test_shadow_dataset = torch.utils.data.random_split(shadow_dataset, [len(shadow_dataset)//2]*2)
        train_shadow_loader = torch.utils.data.DataLoader(train_shadow_dataset,**train_kwargs)
        test_shadow_loader = torch.utils.data.DataLoader(test_shadow_dataset, **train_kwargs)
        train(shadow_model, device, train_shadow_loader, test_shadow_loader, target_epochs)
        torch.save(train_shadow_dataset, "./shadow_train_cifar100.pt")
        torch.save(test_shadow_dataset, "./shadow_test_cifar100.pt")
        torch.save(shadow_model.state_dict(), "./shadow_model_cifar100.pt")
    else:
        shadow_model.load_state_dict(torch.load("./shadow_model_cifar100.pt"))
        train_shadow_dataset = torch.load("./shadow_train_cifar100.pt")
        test_shadow_dataset = torch.load("./shadow_test_cifar100.pt")
        train_shadow_loader = torch.utils.data.DataLoader(train_shadow_dataset,**train_kwargs)
        test_shadow_loader = torch.utils.data.DataLoader(test_shadow_dataset, **train_kwargs)

    shadow_model.eval()

    # IN
    preds_IN = predict_( shadow_model, device, train_shadow_loader )
    labels_IN = np.ones(8000)

    # OUT
    preds_OUT = predict_( shadow_model, device, test_shadow_loader )
    labels_OUT = np.zeros(8000)

    # ALL = IN + OUT
    preds = np.concatenate([preds_IN, preds_OUT])
    labels = np.concatenate([labels_IN, labels_OUT])

    # Top 3 prediction values for every data point
    preds = getTopk(torch.from_numpy(preds), 3).numpy()

    # Create attack dataset
    attack_dataset = torch.utils.data.TensorDataset(torch.Tensor(preds).type(torch.FloatTensor), torch.Tensor(labels).type(torch.LongTensor) )
    train_attacker_loader = torch.utils.data.DataLoader(attack_dataset, **{'batch_size': 32, "shuffle": True})

    # Train classifier
    attack_model = Classifier().to(device)
    train_classifier(attack_model, device, train_attacker_loader, classifier_epochs)

    attack_model.eval()

    # Evaluation
    train_target_loader, test_target_loader = target_loaders
    correct = 0
    with torch.no_grad():
        for data, target in train_target_loader:
            data, target = data.to(device), target.to(device)
            output = getTopk( torch.nn.functional.softmax(target_model(data), dim=1), 3 )
            output = attack_model(output)
            output = output.round()
            correct += (output == 1).sum().item()
        for data, target in test_target_loader:
            data, target = data.to(device), target.to(device)
            output = getTopk( torch.nn.functional.softmax(target_model(data), dim=1), 3 )
            output = attack_model(output)
            output = output.round()
            correct += (output == 0).sum().item()
    
    return (correct / 16000) * 100
