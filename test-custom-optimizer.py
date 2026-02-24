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
class CustomMSELoss(nn.Module):
    def forward(self, preds, targets):
        return ((preds - targets) ** 2).mean()

#training
model = nn.Linear(1, 1)
optimizer = CustomSGD(model.parameters(), lr=0.1)
criterion = CustomMSELoss()
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])
for epoch in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")