import torch
import numpy as np
import torch.nn.functional as F


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def cw_whitebox(model, X, y, normalize, epsilon=8, alpha=2,
                 attack_iters=30, restarts=1):

    epsilon = epsilon / 255.0
    alpha = alpha / 255.0
    model.eval()
    for _ in range(restarts):
        x_adv = X.detach() + torch.empty_like(X).uniform_(-epsilon, epsilon).cuda().detach()
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()

        for _ in range(attack_iters):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = CW_loss(model(normalize(x_adv)), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv



