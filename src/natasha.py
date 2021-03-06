import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from utils import param_norm, param_normalize, hessian_vector
from spider_boost import spider_boost


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
        W.append(param_normalize(w_1))

        #Perform needed amount of Oja iterations
        for i in range(1, T):
            w_last = W[-1]
            prod = hessian_vector(w_last, model, loss_fn, regularizer, dl_1)
            w = [l - eta / L * p for l, p in zip(w_last, prod)]
            W.append(param_normalize(w))
        
        #Get a candidate for eigenvector
        eigvec = W[torch.randint(T, (1, ))]

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
               n_subepochs=None):
    total_loss = np.zeros(n_epochs)

    if regularizer is None:

        def regularizer(x):
            return 0

    dl_1 = DataLoader(train_dataset, 1, shuffle=True)
    dl_B = DataLoader(train_dataset, batch_size, shuffle=True)

    dl_full = DataLoader(train_dataset, len(train_dataset))
    A, b = next(iter(dl_full))

    if n_subepochs is None:
        n_subepochs = int(batch_size**0.5)

    for epoch in range(n_epochs):

        x_B, y_B = next(iter(dl_B))

        model_tilde = copy.deepcopy(model)
        if torch.cuda.is_available():
            model_tilde = model_tilde.cuda()

        y_B_pred = model(x_B)
        loss = loss_fn(y_B_pred, y_B) + regularizer(model.parameters())
        mu_s = torch.autograd.grad(loss, model.parameters())

        #Each epoch divides into subepochs
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

            #Next point is an average of X elements
            x_caps = tuple(
                map(lambda x: torch.mean(x, dim=0),
                    list(map(torch.stack, zip(*X))))) 

            with torch.no_grad():
                for p, x in zip(model.parameters(), x_caps):
                    p.copy_(x)

        with torch.no_grad():
            b_pred = model(A)
            full_loss = loss_fn(b_pred, b) + \
                regularizer(model.parameters())
            total_loss[epoch] = full_loss.item()
    return total_loss


def natasha_reg(parameters, init_parameters, L, L_2, delta):
    dist = param_norm([p - init for p, init in zip(parameters, init_parameters)])
    return L * torch.nn.functional.relu(dist - delta/L_2)**2

def natasha_2(train_dataset,
              batch_size,
              model,
              loss_fn,
              regularizer,
              lr,
              n_epochs,
              spider=False,
              oja_iterations=10,
              L=1,
              L_2=1000):
    total_loss = np.zeros(n_epochs)

    if regularizer is None:

        def regularizer(x):
            return 0

    delta = batch_size**(-0.125)
    B = batch_size
    T = oja_iterations

    dl_full = DataLoader(train_dataset, len(train_dataset))
    A, b = next(iter(dl_full))

    epoch = 0
    while epoch < n_epochs:
        
        #Compute minimal eigenvalue via Oja's algorythm
        eigvecs, eigval = oja_eigenthings(model,
                                          loss_fn,
                                          regularizer,
                                          train_dataset,
                                          T,
                                          L=L)
        #Choose whether to perform an optimizer step or try to move from saddle point
        if eigval <= -0.5 * delta:
            # +/- 1 with p=0.5
            factor = 2 * torch.bernoulli(torch.tensor(0.5)) - 1
            with torch.no_grad():
                for p, ev in zip(model.parameters(), eigvecs):
                    p += factor*delta / L_2 * ev
        else:
            curr_params = tuple([p.clone() for p in model.parameters()])

            def reg_k(x):
                return natasha_reg(x, curr_params, L, L_2,
                                   delta) + regularizer(x)

            if spider:
                _ = spider_boost(train_dataset, B, model, loss_fn, reg_k, lr,
                                 1)
            else:
                _ = natasha_15(train_dataset,
                               B,
                               model,
                               loss_fn,
                               reg_k,
                               lr,
                               1,
                               sigma=3 * delta)

            with torch.no_grad():
                b_pred = model(A)
                loss = loss_fn(b_pred, b) + regularizer(model.parameters())
                total_loss[epoch] = loss.item()
            epoch += 1
    return total_loss
