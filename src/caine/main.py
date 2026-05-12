import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class CaineVisualizer:
    def __init__(self):
        self.model = nn.Linear(1, 1)
        self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.y = -5 * self.x + 0.1 * torch.randn(self.x.size())

    def run_training(self, epochs):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        for e in range(epochs):
            y_pred = self.model(self.x)
            loss = criterion(y_pred, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return {"loss": loss.item(), "epochs": epochs, "trend": "stable"}

    def visualize(self):
        plt.show()
