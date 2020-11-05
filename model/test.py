import torch

a = torch.tensor([[[1, 2, 3], [1, 2, 4]], [[1, 7, 3], [1, 2, 4]]])
print(a.sum(dim=-2))
