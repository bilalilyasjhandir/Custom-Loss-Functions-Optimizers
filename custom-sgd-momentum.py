import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
class CustomSGDMomentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(param.data)
                velocity = state["velocity"]
                velocity.mul_(momentum).add_(param.grad.data)
                param.data -= lr * velocity

#test
model = nn.Linear(2, 1)
optimizer = CustomSGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])
criterion = nn.MSELoss()
print("Initial weights:", model.weight.data)
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
print("Updated weights:", model.weight.data)