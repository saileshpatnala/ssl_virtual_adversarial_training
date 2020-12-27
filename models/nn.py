import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import get_device
from sklearn.metrics import accuracy_score


class NeuralNet(nn.Module):
    def __init__(self, input_size, out_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, out_size+1)
        self.device = get_device()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


def train_nn(X, y, model, optimizer,
             num_epochs=10,
             loss_fn=nn.CrossEntropyLoss()):
    for i in range(num_epochs):
        print(f"Epoch: {i}")
        out = model.forward(X)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = accuracy_score(y.data.cpu().numpy(), np.argmax(out.data.cpu().numpy(), axis=1))
        print("Accuracy %f:" % accuracy)
        print("---------------------------------")
