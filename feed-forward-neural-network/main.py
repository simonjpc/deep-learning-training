import sys
import torch
from typing import List, Tuple
import numpy as np
from data_handler import load_dataset, preprocess_data
from utils import batcher
from model import NeuralNetwork, cross_entropy, backward_pass, optimizer_step


def training():
    try:
        nb_epochs, alpha = sys.argv[1], sys.argv[2]
    except:
        nb_epochs, alpha = 10, 1e-3
        print(
            "`nb_epochs` and `alpha` not specified, using default values nb_epochs=10 & alpha=1e-3"
        )

    (x_train, y_train), (x_test, y_test) = load_dataset()
    train_batches = batcher((x_train, y_train))
    test_batches = batcher((x_test, y_test))

    nn = NeuralNetwork([784, 256, 64, 10])
    for epoch in range(nb_epochs):
        print(f"epoch {epoch + 1} / {nb_epochs}")
        avg_train_loss = 0
        for xtrain, ytrain in train_batches:

            xtrain, ytrain = preprocess_data(xtrain, ytrain)
            logits = nn.forward(xtrain)
            softmax, loss = cross_entropy(logits, ytrain)
            avg_train_loss += loss
            grads = backward_pass(ytrain, softmax, nn.parameters, nn.activations)
            nn.parameters = optimizer_step(nn.parameters, grads, alpha)

        avg_train_loss = avg_train_loss / len(train_batches)

        avg_test_loss = 0
        for xtest, ytest in test_batches:

            xtest, ytest = preprocess_data(xtest, ytest)
            logits = nn.forward(xtest)
            softmax, test_loss = cross_entropy(logits, ytest)
            avg_test_loss += test_loss
            grads = backward_pass(ytest, softmax, nn.parameters, nn.activations)
            nn.parameters = optimizer_step(nn.parameters, grads, alpha)

        avg_test_loss = avg_test_loss / len(test_batches)
        print(f"\tavg training loss: {avg_train_loss}")
        print(f"\tavg test loss: {avg_test_loss}")


if __name__ == "__main__":

    training()
