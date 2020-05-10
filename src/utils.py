import numpy as np
import torch
import sys
from torch.utils.data import DataLoader, Dataset


def gradient_descent(train_dataset, batch_size, model, loss, regularizer, lr, n_epochs):
    total_loss = np.zeros(n_epochs)

    x_full, y_full = train_dataset.data, train_dataset.targets
    batch_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            print(f'epoch: {epoch}', file=sys.stderr)

        for x_batch, y_batch in batch_train_loader:
            # Optimize with batch gradient
            opt.zero_grad()
            y_pred = model(x_batch)
            batch_loss = loss(y_pred, y_batch) + regularizer(model.parameters())
            batch_loss.backward()
            opt.step()

        # Save loss at the end of the epoch
        y_pred = model(x_full)
        total_loss[epoch] = loss(y_pred, y_full).item() + regularizer(model.parameters()).item()
    return total_loss


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.data, self.targets = dataset.data, dataset.targets

        self.data = self.data.float().view(len(self.data), -1)
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.targets = self.targets.cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def regularizer(parameters, alpha=0.1):
    value = 0.0
    for p in parameters:
        value += alpha * (p ** 2 / (1 + p ** 2)).sum()
    return value
