import torch
import torch.nn as nn
class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, predictions, targets):
        loss = (predictions - targets) ** 2
        return loss.mean()

#test
predictions = torch.tensor([2.5, 0.0, 2.1], requires_grad=True)
targets = torch.tensor([3.0, -0.5, 2.0])
loss_fn = CustomMSELoss()
loss = loss_fn(predictions, targets)
loss.backward()
print("Custom MSE Loss:", loss.item())
print("Gradients:", predictions.grad)