import torch

a = torch.tensor([1, 2, 3])

print(1 if torch.argmax(a) == 2 else 0)
