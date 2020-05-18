import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from utils import param_norm, hessian_vector
import sys


def oja_eigenthings(model,
                    loss_fn,
                    regularizer,
                    train_dataset,
                    n_iterations,
                    p=1e-3,
                    L=1000):
    """
    Computing estimates for minimum eigenvalue and its eigenvector
    """
    Eig = []

    T = n_iterations

    dl_1 = DataLoader(train_dataset, 1, shuffle=True)
    dl_T = DataLoader(train_dataset, T, shuffle=True)

    eta = np.sqrt(T)

    for _ in range(int(-1 * np.log(p))):
        W = []
        w_1 = [
            torch.zeros_like(p).normal_(mean=0, std=1)
            for p in model.parameters()
        ]
        W.append(tuple(el / param_norm(w_1) for el in w_1))
        for i in range(1, T):
            w_last = W[-1]
            prod = hessian_vector(w_last, model, loss_fn, regularizer, dl_1)
            w = [l - eta / L * p for l, p in zip(w_last, prod)]
            W.append(tuple(el / param_norm(w) for el in w))

        eigvec = W[torch.randint(T, (1,))]  # candidate for eigenvector

        prod = hessian_vector(eigvec, model, loss_fn, regularizer, dl_T)
        eigval = torch.zeros(1)
        for v, p in zip(eigvec, prod):
            eigval += (v * p).sum()

        Eig.append((eigvec, eigval))

    Eig.sort(key=lambda x: x[1])
    return Eig[0]


def natasha_15(train_dataset,
               batch_size,
               model,
               loss_fn,
               regularizer,
               lr,
               n_epochs,
               sigma,
               n_subepochs=None,
               loss_log=True):
    total_loss = np.zeros(n_epochs)

    if regularizer is None:

        def regularizer(x):
            return 0

    dl_1 = DataLoader(train_dataset, 1, shuffle=True)
    dl_B = DataLoader(train_dataset, batch_size, shuffle=True)
    if loss_log:
        dl_full = DataLoader(train_dataset, len(train_dataset))
        A, b = next(iter(dl_full))

    if n_subepochs is None:
        n_subepochs = int(batch_size**0.5)

    for epoch, (x_B, y_B) in enumerate(dl_B):
        if epoch >= n_epochs:
            break

        model_tilde = copy.deepcopy(model)
        if torch.cuda.is_available():
            model_tilde = model_tilde.cuda()

        y_B_pred = model(x_B)
        loss = loss_fn(y_B_pred, y_B) + regularizer(model.parameters())
        mu_s = torch.autograd.grad(loss, model.parameters())

        for subepoch in range(n_subepochs):
            x_0 = tuple([p.detach() for p in model.parameters()])
            X = [x_0]
            m = max(int(batch_size / n_subepochs), 1)
            for t, (x, y) in enumerate(dl_1):
                if t >= m:
                    break
                y_pred_tilde = model_tilde(x)
                loss_tilde = loss_fn(y_pred_tilde, y) + \
                    regularizer(model_tilde.parameters())
                grads_tilde = torch.autograd.grad(loss_tilde,
                                                  model_tilde.parameters())
                y_pred_t = model(x)
                loss_t = loss_fn(y_pred_t, y) + \
                    regularizer(model.parameters())
                grads_t = torch.autograd.grad(loss_t, model.parameters())
                nablas = tuple([
                    n_t - n_til + mu + 2 * sigma * (x_t - x_cap)
                    for n_t, n_til, mu, x_t, x_cap in zip(
                        grads_t, grads_tilde, mu_s, X[-1], x_0)
                ])
                with torch.no_grad():
                    for p, nabla in zip(model.parameters(), nablas):
                        p -= lr * nabla
                X.append(tuple([p.detach() for p in model.parameters()]))

            x_caps = tuple(
                map(lambda x: torch.mean(x, dim=0),
                    list(map(torch.stack, zip(*X)))))
            with torch.no_grad():
                for p, x in zip(model.parameters(), x_caps):
                    p.copy_(x)

        if loss_log:
            with torch.no_grad():
                b_pred = model(A)
                full_loss = loss_fn(b_pred, b) + \
                    regularizer(model.parameters())
                total_loss[epoch] = full_loss.item()
    return total_loss


def natasha_reg(parameters, init_parameters, L, L_2, delta):
    diff = [p - init for p, init in zip(parameters, init_parameters)]
    return L * (max(0, param_norm(diff) - delta / L_2))**2


def natasha_2(train_dataset,
              batch_size,
              model,
              loss_fn,
              regularizer,
              lr,
              n_epochs,
              natasha15_epochs=1,
              oja_iterations=10,
              L=100,
              L_2=10):
    total_loss = np.zeros(n_epochs)

    if regularizer is None:

        def regularizer(x):
            return 0

    delta = batch_size**(-0.125)
    B = batch_size  # min(len(train_dataset), int(n_epochs**1.6))
    T = oja_iterations  # max(2, int(n_epochs**0.4))

    dl_full = DataLoader(train_dataset, len(train_dataset))
    A, b = next(iter(dl_full))

    epoch = 0
    while epoch < n_epochs:

        eigvecs, eigval = oja_eigenthings(model,
                                          loss_fn,
                                          regularizer,
                                          train_dataset,
                                          T,
                                          L=L)
        if eigval <= -0.5 * delta:
            # +/- 1 with p=0.5
            factor = 2 * torch.bernoulli(torch.tensor(0.5)) - 1
            with torch.no_grad():
                for p, ev in zip(model.parameters(), eigvecs):
                    p += factor / L_2 * ev
        else:
            curr_params = tuple([p.clone() for p in model.parameters()])

            def reg_k(x):
                return natasha_reg(x, curr_params, L, L_2,
                                   delta) + regularizer(x)

            _ = natasha_15(train_dataset,
                       B,
                       model,
                       loss_fn,
                       reg_k,
                       lr,
                       natasha15_epochs,
                       sigma=3 * delta,
                       loss_log=False)
            with torch.no_grad():
                b_pred = model(A)
                loss = loss_fn(b_pred, b) + regularizer(model.parameters())
                total_loss[epoch] = loss.item()
            epoch += 1
    return total_loss
