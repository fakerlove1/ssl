def interleave(x, size):
    s = list(x.shape)
    print([-1] + s[1:])
    print([-1, size] + s[1:])
    print(x.reshape([-1, size] + s[1:]).shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


import torch

mu = 7

x = torch.randn(15, 3, 32, 32)

y = interleave(x, 2 * mu + 1)
print(y.shape)
