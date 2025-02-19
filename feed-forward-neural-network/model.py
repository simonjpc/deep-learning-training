import torch
from typing import List, Tuple


class NeuralNetwork:

    def __init__(self, sizes: List[int]):

        self.sizes = sizes
        self.parameters = {}
        self.activations = {}

        for l in range(1, len(sizes)):
            self.parameters[f"W{l}"] = (
                torch.randn(sizes[l - 1], sizes[l], dtype=torch.float32) * 0.01
            )
            self.parameters[f"b{l}"] = torch.randn(1, sizes[l], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.activations["A0"] = x

        for l in range(1, len(self.sizes)):
            Z = (
                torch.matmul(self.activations[f"A{l-1}"], self.parameters[f"W{l}"])
                + self.parameters[f"b{l}"]
            )
            A = torch.relu(Z)
            self.activations[f"Z{l}"], self.activations[f"A{l}"] = Z, A

        self.activations.pop(f"A{len(self.sizes) - 1}")

        return Z


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor]:

    reg_exp = torch.exp(logits - torch.max(logits))
    softmax = reg_exp / torch.sum(reg_exp, dim=1, keepdim=True)
    loss = -torch.sum(labels * torch.log(softmax) + 1e-9)
    return softmax, loss


def backward_pass(y, softmax, parameters, activations):

    dZ = softmax - y

    grads = {}
    for l in reversed(range(1, len(parameters) // 2 + 1)):

        grads[f"dW{l}"] = torch.matmul(activations[f"A{l-1}"].T, dZ)
        grads[f"db{l}"] = torch.sum(dZ, dim=0, keepdim=True)

        if l > 1:
            dA = torch.matmul(dZ, parameters[f"W{l}"].T)
            dZ = dA * (activations[f"A{l-1}"] > 0)

    return grads


def optimizer_step(parameters, grads, alpha):

    for l in range(1, len(parameters) // 2 + 1):
        parameters[f"W{l}"] -= alpha * grads[f"dW{l}"]
        parameters[f"b{l}"] -= alpha * grads[f"db{l}"]

    return parameters
