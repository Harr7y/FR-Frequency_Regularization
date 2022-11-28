import torch
import torch.nn.functional as F
import torch.nn as nn


def pgd_whitebox(model, X, y, normalize, epsilon=8, alpha=2,
                 attack_iters=20, restarts=1):

    epsilon = epsilon / 255.0
    alpha = alpha / 255.0
    model.eval()
    for _ in range(restarts):
        x_adv = X.detach() + torch.empty_like(X).uniform_(-epsilon, epsilon).cuda().detach()
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()

        for _ in range(attack_iters):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(normalize(x_adv)), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv
