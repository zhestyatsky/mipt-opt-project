import torch

def param_norm(params):
    s = torch.zeros(1)
    if torch.cuda.is_available():
        s = s.cuda()
    for param in params:
        s += (param**2).sum()
    return torch.sqrt(s)
    
def hessian_vector(vec, model, loss_fn, regularizer, dataloader):
    #w is a tuple same size as model.parameters()
    A_i, b_i = next(iter(dataloader))
    b_pred = model(A_i)
    loss = loss_fn(b_i, b_pred) + regularizer(model.parameters())

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    scal = torch.zeros(1)
    if torch.cuda.is_available():
        scal = scal.cuda()
    for _w, grad in zip(vec, grads):
        scal += (_w*grad).sum()

    prod = torch.autograd.grad(scal, model.parameters())

    return prod

def regularizer(parameters, alpha=0.1):
    value = 0.0
    for p in parameters:
        value += alpha * (p ** 2 / (1 + p ** 2)).sum()
    return value
