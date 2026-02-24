import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
class CustomSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.data -= lr * param.grad.data

#test
model = nn.Linear(2, 1)
optimizer = CustomSGD(model.parameters(), lr=0.1)
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])
criterion = nn.MSELoss()
print("Initial weights:", model.weight.data)
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
print("Updated weights:", model.weight.data)