import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from utils import param_norm, hessian_vector

def oja_eigenthings(model, loss_fn, regularizer, train_dataset, n_iterations, p = 1e-3, L = 1000): 
    '''
    Computing etimates for minimum eigenvalue and its eigenvector
    '''
    Eig = []
    
    T = n_iterations

    dl_1 = DataLoader(train_dataset, 1, shuffle=True)
    dl_T = DataLoader(train_dataset, T, shuffle=True)

    eta = np.sqrt(T)
     
    
    for _ in range(int(-1*np.log(p))): 
        W = []
        w_1 = [torch.zeros_like(p).normal_(mean=0,std=1) for p in model.parameters()]
        W.append(tuple( el/param_norm(w_1) for el in w_1))
        for i in range(1, T):
            w_last = W[-1]
            prod = hessian_vector(w_last, model, loss_fn, regularizer, dl_1)
            w = [l - eta/L*p for l, p in zip(w_last, prod)]
            W.append(tuple( el/param_norm(w) for el in w))
            
        eigvec = W[torch.randint(T, (1,))] #candidate for eigenvector

        prod = hessian_vector(eigvec, model, loss_fn, regularizer, dl_T)
        eigval = torch.zeros(1)
        for v,p in zip(eigvec, prod):
            eigval += (v*p).sum()

        Eig.append((eigvec, eigval))

    Eig.sort(key = lambda x : x[1])
    return Eig[0]

def natasha_15(train_dataset, batch_size, model, loss_fn, regularizer, lr, n_epochs, sigma, loss_log=True):
    total_loss = np.zeros(n_epochs)

    if regularizer is None:
        regularizer = lambda x : 0

    dl_1 = DataLoader(train_dataset, 1, shuffle=True)
    dl_B = DataLoader(train_dataset, batch_size, shuffle=True)
    if loss_log:
        dl_full = DataLoader(train_dataset,len(train_dataset))
        A, b = next(iter(dl_full))
        #with torch.no_grad():
        #    b_pred = model(A)
        #    full_loss = loss_fn(b, b_pred) +  regularizer(model.parameters())
        #    total_loss[0] = full_loss.item()
        

    n_subepochs = max(int(lr*sigma*batch_size), 1)

    for epoch in range(n_epochs):

        model_tilde = copy.deepcopy(model)
        if torch.cuda.is_available():
            model_tilde = model_tilde.cuda()

        A_i, b_i = next(iter(dl_B))
        b_pred = model(A_i)
        loss = loss_fn(b_pred, b_i) + regularizer(model.parameters())
        mu_s = torch.autograd.grad(loss, model.parameters())
        
        for subepoch in range(n_subepochs):
            x_0 = tuple([p.detach() for p in model.parameters()])
            X = [x_0]
            m = max(int(batch_size/n_subepochs),1)
            for t in range(m):
                A_i, b_i = next(iter(dl_1))
                b_pred_tilde = model_tilde(A_i)
                loss_tilde = loss_fn(b_pred_tilde, b_i) + regularizer(model_tilde.parameters())
                grads_tilde = torch.autograd.grad(loss_tilde, model_tilde.parameters())
                b_pred_t = model(A_i)
                loss_t = loss_fn(b_pred_t, b_i) + regularizer(model.parameters())
                grads_t = torch.autograd.grad(loss_t, model.parameters())
                nablas = tuple([n_t - n_til + mu + 2*sigma*(x_t - x_cap)  for n_t, n_til, mu, x_t, x_cap in zip(grads_t, grads_tilde, mu_s, X[-1], x_0)])
                with torch.no_grad():
                    for p, nabla in zip(model.parameters(), nablas):
                        p -= lr*nabla
                X.append(tuple([p.detach() for p in model.parameters()]))
            ind = torch.randint(m+1, (1,))
            # print(ind)
            x_caps = X[ind]
            with torch.no_grad():
                for p, x in zip(model.parameters(), x_caps):
                    p.copy_(x)

        if loss_log:
            with torch.no_grad():
                b_pred = model(A)
                full_loss = loss_fn(b_pred, b) +  regularizer(model.parameters())
                total_loss[epoch] = full_loss.item()
    return total_loss

def natasha_reg(parameters, init_parameters, L, L_2, delta):
    diff = [p-init for p, init in zip(parameters, init_parameters)]
    return L * (max(0, param_norm(diff) - delta/L_2 ))**2


def natasha_2(train_dataset, model, loss_fn, regularizer, lr, n_epochs, L_2 = 1):
    total_loss = np.zeros(n_epochs)

    if regularizer is None:
        regularizer = lambda x : 0

    delta = n_epochs**(-0.2)
    L = 1.0/lr
    B = min(len(train_dataset), int(n_epochs**1.6))
    T = max(2, int(n_epochs**0.4))

    dl_full = DataLoader(train_dataset,len(train_dataset))
    A, b = next(iter(dl_full))

    #with torch.no_grad():
    #  b_pred = model(A)
    #  loss = loss_fn(b, b_pred) + regularizer(model.parameters())
    #  total_loss[0] = loss.item()

    epoch = 0
    while(epoch < n_epochs):
        eigvecs, eigval = oja_eigenthings(model, loss_fn, regularizer, train_dataset, T, L = L)
        if(eigval <= -0.5*delta):
            factor = 2*torch.bernoulli(torch.tensor(0.5))-1 # +/- 1 with p=0.5
            with torch.no_grad():
                for p, ev in zip(model.parameters(), eigvecs):
                    p += factor/L_2*ev
        else:
            curr_params = tuple([p.clone() for p in model.parameters()])
            reg_k = lambda x : natasha_reg(x, curr_params, L, L_2, delta) + regularizer(x)

            natasha_1_5(train_dataset, B, model, loss_fn, reg_k, lr/5, 1, 3*delta)
            with torch.no_grad():
                b_pred = model(A)
                loss = loss_fn(b_pred, b) + regularizer(model.parameters())
                total_loss[epoch] = loss.item()
            epoch += 1
    return total_loss

                      

