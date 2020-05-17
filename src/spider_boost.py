import copy
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader


def spider_boost(train_dataset, batch_size, model,
                 loss, regularizer, lr, n_epochs):
    total_loss = np.zeros(n_epochs)

    x_full, y_full = train_dataset.data, train_dataset.targets
    batch_train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

    model_prev = copy.deepcopy(model)
    if torch.cuda.is_available():
        model_prev = model_prev.cuda()

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt_prev = torch.optim.SGD(model_prev.parameters(), lr=lr)

    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            print(f'epoch: {epoch}', file=sys.stderr)

        # Calculate full gradient first
        opt.zero_grad()
        y_pred = model(x_full)
        main_loss = loss(y_pred, y_full) + regularizer(model.parameters())
        main_loss.backward()

        # Saving current model weights and zero grad them
        for param, param_prev in zip(
                model.parameters(), model_prev.parameters()):
            param_prev.data = param.data.clone().detach()
        opt_prev.zero_grad()

        # Optimize
        opt.step()

        for x_batch, y_batch in batch_train_loader:
            # Add current state gradients
            y_pred = model(x_batch)
            batch_loss = loss(y_pred, y_batch) + \
                regularizer(model.parameters())
            batch_loss.backward()

            # Calculate previous state gradients
            y_pred_prev = model_prev(x_batch)
            batch_loss_prev = loss(y_pred_prev, y_batch)
            batch_loss_prev.backward()

            # Subtract previous state gradients
            for param, param_prev in zip(
                    model.parameters(), model_prev.parameters()):
                param.grad.data -= param_prev.grad.data
                param_prev.data = param.data.clone().detach()
            opt_prev.zero_grad()

            # Optimize
            opt.step()

        # Save loss at the end of the epoch
        y_pred = model(x_full)
        total_loss[epoch] = loss(y_pred, y_full).item(
        ) + regularizer(model.parameters()).item()
    return total_loss
