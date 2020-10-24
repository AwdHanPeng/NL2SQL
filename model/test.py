import torch

a = torch.tensor([1, 2, 3])
b = a.masked_fill(torch.tensor([0, 0, 1]).gt(0), 5)
print(b)
