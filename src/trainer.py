import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.model import NeuralNetwork
from src.function_distrib import FunctionDistrib
from src.utils import compute_wasserstein_distance


class Trainer:
    def __init__(self, config, function_distrib=None):
        self.config = config

        torch.manual_seed(self.config.get("seed", 42))

        self.function_distrib = function_distrib
        if function_distrib is None:
            self.function_distrib = FunctionDistrib(**self.config["function_distrib"])

        self.load_model()
        self.load_optimizer()
        self.load_criterion()

    def load_model(self):
        self.config["model"]["input_size"] = self.config["model"].get(
            "input_size", self.function_distrib.d
        )
        self.config["model"]["output_size"] = self.config["model"].get(
            "output_size", self.function_distrib.Y.shape[1]
        )
        self.model = NeuralNetwork(**self.config["model"])

    def load_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), **self.config["optimizer"])

    def load_criterion(self):
        self.criterion = nn.MSELoss()

    def train(self, plot=False):
        losses = []
        w2_distances = []

        for epoch in range(self.config.get("epochs", 300)):
            Y, Z = self.function_distrib.resample()
            self.optimizer.zero_grad()
            Y_pred = self.model(torch.from_numpy(Z).float())
            loss = self.criterion(Y_pred, torch.from_numpy(Y).float())
            losses.append(loss.item())
            w2_distances.append(compute_wasserstein_distance(Y, Y_pred))
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0 and plot:
                print("Epoch: {}, loss: {}".format(epoch, loss.item()))

        if plot:
            plt.plot(losses)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.show()

            plt.plot(w2_distances)
            plt.ylabel("Wasserstein distance")
            plt.xlabel("Epoch")
            plt.show()

        # We evaluate the Neural Network
        Y_pred = self.model(torch.from_numpy(Z).float())
        loss = self.criterion(Y_pred, torch.from_numpy(Y).float())
        w2 = compute_wasserstein_distance(Y, Y_pred)

        if plot:
            print("Final loss: {}".format(loss.item()))
            print("Final Wasserstein distance: {}".format(w2))

        return losses, w2_distances
