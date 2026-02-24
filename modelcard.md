# 📘 Custom Training Components in PyTorch
### Loss Functions and Optimizers – Educational Implementations

---

## 📌 Overview

This repository contains custom implementations of fundamental training components in [PyTorch](https://pytorch.org/), including:

- Cross-Entropy Loss
- Focal Loss
- Mean Squared Error (MSE) Loss
- Stochastic Gradient Descent (SGD)
- SGD with Momentum

These implementations were developed for educational purposes to deeply understand:

- Autograd mechanics
- Gradient flow
- Optimizer state management
- Loss function mathematics
- Class imbalance handling
- Training loop construction

---

## 🧠 Implemented Components

---

### 1️⃣ Custom Cross-Entropy Loss

**Type:** Multi-class classification loss  

#### Mathematical Form:

L = -(1/N) * Σ log(p_y)

Where:
- p_y is the predicted probability of the true class.
- `log_softmax` is used for numerical stability.

#### Key Design Decisions:
- Uses `torch.log_softmax` instead of `softmax + log`
- Index-based class selection
- Mean reduction

#### Use Case:
- Standard multi-class classification problems

---

### 2️⃣ Custom Focal Loss

**Type:** Imbalance-aware classification loss  

#### Mathematical Form:

L = - (1 - p_t)^γ * log(p_t)

Where:
- p_t = probability of true class
- γ (gamma) = focusing parameter

#### Purpose:
- Down-weights easy examples
- Focuses training on hard samples

#### Key Feature:
- Adjustable `gamma` parameter
- Built using log-softmax for stability

#### Use Case:
- Imbalanced datasets
- Object detection
- Rare-class classification

---

### 3️⃣ Custom MSE Loss

**Type:** Regression loss  

#### Formula:

L = (1/N) * Σ (y_pred - y_true)^2

#### Purpose:
- Measures squared error between predictions and targets
- Used in regression tasks

---

## ⚙️ Custom Optimizers

---

### 4️⃣ Custom SGD

#### Update Rule:

θ = θ - η * ∇L

Where:
- η = learning rate

#### Implementation Highlights:
- Extends `torch.optim.Optimizer`
- Iterates over `param_groups`
- Performs direct gradient-based updates

#### Use Case:
- Baseline optimization
- Understanding gradient descent mechanics

---

### 5️⃣ Custom SGD with Momentum

#### Update Rule:

v_t = μ * v_(t-1) + ∇L  
θ = θ - η * v_t

Where:
- μ = momentum coefficient
- v_t = velocity

#### Implementation Highlights:
- Maintains per-parameter state dictionary
- Stores velocity tensor
- Demonstrates optimizer state handling

#### Advantages Over Vanilla SGD:
- Faster convergence
- Reduced oscillation
- Smoother optimization trajectory

---

## 🧪 Experimental Validation

The components were tested using:

- Synthetic classification logits
- Small regression datasets
- Simple neural network training loops
- Gradient verification using `.backward()`

Example:
- Linear regression trained with Custom SGD
- Observed decreasing loss over epochs
- Verified gradient propagation

---

## 🔍 Comparison to Native PyTorch

| Component | Custom Version | PyTorch Equivalent |
|------------|----------------|--------------------|
| Cross Entropy | `CustomCrossEntropyLoss` | `nn.CrossEntropyLoss` |
| Focal Loss | `FocalLoss` | Not built-in (custom required) |
| MSE | `CustomMSELoss` | `nn.MSELoss` |
| SGD | `CustomSGD` | `torch.optim.SGD` |
| SGD + Momentum | `CustomSGDMomentum` | `torch.optim.SGD(momentum=...)` |

---

## 🎯 Design Philosophy

These implementations prioritize:

- Educational clarity over production efficiency
- Explicit tensor operations
- Understanding gradient computation
- Demonstrating how optimizers maintain internal state

---

## ⚠️ Limitations

- No support for:
  - Weight decay
  - Nesterov momentum
  - Mixed precision training
  - Distributed training
- No label smoothing in cross-entropy
- No alpha-balancing in focal loss
- Minimal error handling

These are simplified implementations intended for learning.

---

## 📊 When to Use Each Component

| Scenario | Recommended Component |
|----------|-----------------------|
| Multi-class classification | Custom Cross Entropy |
| Imbalanced classification | Focal Loss |
| Regression | Custom MSE |
| Simple baseline training | Custom SGD |
| Faster convergence | SGD with Momentum |

---

## 🎓 Educational Value

This project demonstrates understanding of:

- Autograd mechanics
- Loss derivation
- Numerical stability practices
- Optimizer state management
- Parameter group iteration
- Training loop construction

---

## ⚖️ Ethical Considerations

- Focal Loss may amplify bias if minority classes are poorly represented.
- Improper hyperparameter tuning can destabilize training.
- Not recommended for safety-critical systems without validation.

---

## 🚀 Intended Use

This repository is intended for:

- Educational purposes
- Deep learning practice
- Internship or academic evaluation
- Demonstrating knowledge of training internals

Not intended for direct production deployment without further extension.