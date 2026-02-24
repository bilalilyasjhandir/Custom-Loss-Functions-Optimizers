import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs[range(len(targets)), targets]
        target_probs = probs[range(len(targets)), targets]
        loss = -((1 - target_probs) ** self.gamma) * target_log_probs
        return loss.mean()

#test
logits = torch.tensor([
    [3.0, 0.5, 0.2],
    [0.2, 2.8, 0.1],
    [0.1, 0.2, 3.5]
], requires_grad=True)
targets = torch.tensor([0, 1, 2])
loss_fn = FocalLoss(gamma=2.0)
loss = loss_fn(logits, targets)
loss.backward()
print("Custom Focal Loss:", loss.item())
print("Gradients:\n", logits.grad)