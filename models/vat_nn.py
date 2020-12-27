import torch
import torch.nn as nn
import numpy as np
from models.utils import get_device, l2_normalize, kl_div
from sklearn.metrics import accuracy_score


class VAT_NN(nn.Module):
    def __init__(self, xi=.0001, eps=0.1, ip=2):
        """
        :xi: hyperparameter: small float for finite difference threshold 
        :eps: hyperparameter: value for how much to deviate from original X.
        :ip: value of power iteration for approximation of r_vadv.
        """
        super(VAT_NN, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.device = get_device()
        
    def forward(self, model, x):
        with torch.no_grad():
            pred = model(x)
            
        # random unit tensor for perturbation
        d = torch.randn(x.shape).to(self.device)
        d = l2_normalize(d)
        
        # calculating adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat = model(x + self.xi * d)
            adv_distance = kl_div(pred_hat, pred)
            adv_distance.backward()
            d = l2_normalize(d.grad.data)
            model.zero_grad()
        
        r_adv = d * self.eps
        pred_hat = model(x+r_adv)
        lds = kl_div(pred_hat, pred)
        return lds


def train_vat_nn(dataset, X_labeled_tensor, y_labeled_tensor, model, optimizer, num_epochs=5):
    """
    Function to train VAT-NN model
    """
    i_total_step = 0
    gamma = 4e1
    device = model.device

    for i in range(num_epochs):  # epoch
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)  # batch size
        print(f"Epoch: {i}")
        for u, _ in data_loader:
            i_total_step += 1
            vat_loss = VAT_NN(xi=.0001, eps=.1, ip=1)
            cross_entropy = nn.CrossEntropyLoss()

            lds = vat_loss(model, torch.tensor(u).float().to(device))
            output = model(X_labeled_tensor)
            classification_loss = cross_entropy(output, y_labeled_tensor)
            loss = classification_loss + gamma * lds
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = accuracy_score(y_labeled_tensor.data.cpu().numpy(), np.argmax(output.data.cpu().numpy(), axis=1))
            ce_losses = classification_loss.item()
            vat_losses = gamma * lds.item()

        print("CrossEntropyLoss %f:" % ce_losses)
        print("VATLoss %f:" % vat_losses)
        print("Accuracy %f:" % accuracy)
        print("LDS %f:" % lds)
        print("---------------------------------")
