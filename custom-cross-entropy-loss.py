import torch
import torch.nn as nn
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -log_probs[range(len(targets)), targets]
        return loss.mean()

#test
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
    [1.2, 0.7, 2.1]
], requires_grad=True)
targets = torch.tensor([0, 1, 2])
loss_fn = CustomCrossEntropyLoss()
loss = loss_fn(logits, targets)
loss.backward()
print("Custom Cross-Entropy Loss:", loss.item())
print("Gradients:\n", logits.grad)