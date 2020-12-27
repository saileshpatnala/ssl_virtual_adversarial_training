import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import get_device, save_checkpoint, save_metrics


class LSTM(nn.Module):

    def __init__(self, num_classes, vocab_size, embed_size=300, dimension=128):
        super(LSTM, self).__init__()

        self.device = get_device()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2 * dimension, num_classes + 1)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)

        # pack_padded_sequence had some issues running on gpu
        # https://discuss.pytorch.org/t/pack-padded-sequence-on-gpu/14140
        # https://github.com/pytorch/xla/issues/1522
        packed_input = pack_padded_sequence(text_emb.cpu(), text_len.cpu(), batch_first=True, enforce_sorted=False) \
            .to(self.device)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


METRICS_PATH = '/Users/saileshpatnala/projects/aml/ssl_virtual_adversarial_training'


def train_lstm(train_loader, val_loader, model, optimizer,
               loss_fn=nn.CrossEntropyLoss(),
               num_epochs=5,
               file_path=METRICS_PATH,
               best_val_loss=float("Inf")):
    """
    Function to train LSTM model
    """
    # initialize running values
    running_loss = 0.0
    val_running_loss = 0.0
    global_step = 0

    train_loss_list = []
    val_loss_list = []
    global_steps_list = []

    device = model.device
    eval_every = len(train_loader) // 2

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, (text, text_len)), _ in train_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for (labels, (text, text_len)), _ in val_loader:
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)
                        output = model(text, text_len)

                        loss = loss_fn(output, labels)
                        val_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_val_loss = val_running_loss / len(val_loader)
                train_loss_list.append(average_train_loss)
                val_loss_list.append(average_val_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                val_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_val_loss))

                # checkpoint
                if best_val_loss > average_val_loss:
                    best_val_loss = average_val_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_val_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, val_loss_list, global_steps_list)

    save_metrics(file_path + '/metrics.pt', train_loss_list, val_loss_list, global_steps_list)
    print('Finished Training!')
