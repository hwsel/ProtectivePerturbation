import torch

def tensor_clamp(X):
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1)
    lower_limit = (0.0 - mu) / std
    upper_limit = (1.0 - mu) / std

    lower_limit = lower_limit.to(X.device)
    upper_limit = upper_limit.to(X.device)
    X_clamp = torch.max(torch.min(X, upper_limit), lower_limit)
    return X_clamp


def unnormalize(x):
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1)
    mu = mu.to(x.device)
    std = std.to(x.device)
    x = (x * std) + mu
    return x
