import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.utils import get_device, kl_div, l2_normalize


class VAT_LSTM(nn.Module):
    def __init__(self, xi=.01, eps=0.1, ip=2):
        """
        :xi: hyperparameter: small float for finite difference threshold 
        :eps: hyperparameter: value for how much to deviate from original X.
        :ip: value of power iteration for approximation of r_vadv.
        """
        super(VAT_LSTM, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.device = get_device()

    def _lstm_forward(self, model, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(),
                                        batch_first=True,
                                        enforce_sorted=False).to(self.device)
        packed_output, _ = model.lstm(x_packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), x_len - 1, :model.dimension]
        out_reverse = output[:, 0, model.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = model.drop(out_reduced)

        text_fea1 = model.fc(text_fea)
        text_fea_squeeze = torch.squeeze(text_fea1, 1)
        text_out = torch.sigmoid(text_fea_squeeze)

        return text_out

    def forward(self, model, x, x_len):
        with torch.no_grad():
            pred = model(x, x_len)

        # random unit tensor for perturbation
        d = torch.randn(x.shape).to(self.device)
        d = l2_normalize(d)

        # calculating adversarial direction
        x_embed = model.embedding(x)
        for _ in range(self.ip):
            d.requires_grad_()

            xi_tensor = (torch.zeros(d.shape) + self.xi).to(self.device)
            x_perturb = x_embed + (xi_tensor * d).view(x.shape[0], x.shape[1], 1)
            pred_hat = self._lstm_forward(model, x_perturb, x_len)

            adv_distance = kl_div(pred_hat, pred)
            adv_distance.backward(retain_graph=True)
            d = l2_normalize(d.grad.data)
            model.zero_grad()

        r_adv = (d * self.eps).view(x.shape[0], x.shape[1], 1)
        pred_hat = self._lstm_forward(model, x_embed + r_adv, x_len)
        lds = kl_div(pred_hat, pred)
        return lds


def train_vat_lstm(train_loader, split_loader, model, optimizer,
                   num_epochs=5,
                   eval_every=50):
    """
    Function to train VAT-LSTM model
    """
    i_total_step = 0
    gamma = 4e1

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for (labels, (text, text_len)), _ in train_loader:

            i_total_step += 1
            vat_lstm = VAT_LSTM()
            cross_entropy = nn.CrossEntropyLoss()

            lds = vat_lstm(model, text, text_len)
            labels = torch.zeros(len(split_loader.data()))
            outputs = torch.zeros(size=(len(split_loader.data()), 4))
            for i, ((label_split, (text_split, text_split_len)), _) in enumerate(split_loader):
                batch_size = split_loader.batch_size
                labels[i * batch_size: (i + 1) * batch_size] = label_split
                output = model(text_split, text_split_len)
                outputs[i * batch_size: (i + 1) * batch_size, :] = output

            classification_loss = cross_entropy(outputs, labels.long())
            loss = classification_loss + gamma * lds
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if i_total_step % eval_every == 0:
                ce_loss = classification_loss.item()
                vat_loss = gamma * lds.item()
                print('Cross Entropy Loss: {:.4f}, VAT Loss: {:.4f}, LDS: {:.4f}'
                      .format(ce_loss, vat_loss, lds))

        print("---------------------------------")
