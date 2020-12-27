import os
import torch
import torch.nn.functional as F
import numpy as np
from torchtext.data import Field, TabularDataset, BucketIterator
from sklearn.metrics import classification_report

BIN_DIR = '/Users/saileshpatnala/projects/aml/ssl_virtual_adversarial_training/data/binaries'
DATA_DIR = '/Users/saileshpatnala/projects/aml/ssl_virtual_adversarial_training/data/dbpedia_csv'


def normalize(d):
    d /= (torch.sqrt(torch.sum(d, axis=1)).view(-1, 1) + 1e-16)
    return d


def l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-16
    return d


def kl_div(p, q):
    """
    D_KL(p||q) = Sum(p log p - p log q)
    """
    logp = F.log_softmax(p, dim=1)
    logq = F.log_softmax(q, dim=1)
    p = torch.exp(logp)
    return (p * (logp - logq)).sum(dim=1, keepdim=True).mean()


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    return torch.device(device)


def dbpedia_preprocess_lstm(split=None):
    """
    Function to preprocess data for the LSTM. Involves tokenizing text, building
    the vocabulary and batching the data.
    Returns:
        vocab_size: number of words in the vocabulary
        train_iter: torchtext.data.BucketIterator for training dataset
        valid_iter: torchtext.data.BucketIterator for validation dataset
        test_iter: torchtext.data.BucketIterator for testing dataset
    """
    device = get_device()

    # fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
    text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
    fields = [('label', label_field), ('text', text_field)]

    # tabular dataset for entire dataset
    train, validation, test = TabularDataset.splits(path=os.path.join(DATA_DIR, 'lstm'), train='train.csv',
                                                    validation='validation.csv', test='test.csv', format='CSV',
                                                    fields=fields, skip_header=True)

    train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text),
                                device=device, sort=True, sort_within_batch=True)
    val_iter = BucketIterator(validation, batch_size=32, sort_key=lambda x: len(x.text),
                              device=device, sort=True, sort_within_batch=True)
    test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.text),
                               device=device, sort=True, sort_within_batch=True)

    if split is not None:
        if split == 2:
            train_split, validation_split = TabularDataset.splits(path=DATA_DIR, train='dbpedia_train_split1.csv',
                                                                  validation='dbpedia_validation_split1.csv',
                                                                  format='CSV',
                                                                  fields=fields)

        else:
            train_split, validation_split = TabularDataset.splits(path=DATA_DIR, train='dbpedia_train_split1.csv',
                                                                  validation='dbpedia_validation_split1.csv',
                                                                  format='CSV',
                                                                  fields=fields)
        train_split_iter = BucketIterator(train_split, batch_size=32, sort_key=lambda x: len(x.text),
                                          device=device, sort=True, sort_within_batch=True)
        val_split_iter = BucketIterator(validation_split, batch_size=32, sort_key=lambda x: len(x.text),
                                        device=device, sort=True, sort_within_batch=True)
    else:
        train_split_iter, val_split_iter = None, None

    # build vocabulary
    text_field.build_vocab(train)

    return len(text_field.vocab), train_iter, val_iter, test_iter, train_split_iter, val_split_iter


def save_checkpoint(save_path, model, optimizer, valid_loss):
    """
    Function to save the model checkpoint
    """
    if save_path is None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    """
    Function to load model checkpoint
    """
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=get_device())
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    """
    Function to save model metrics
    """
    if save_path is None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    """
    Function to load model metrics
    """
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=get_device())
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def evaluate_model(model, test_loader, threshold=0.5):
    """
    Function to evaluate model. Prints out classification report.
    """
    y_pred = []
    y_true = []

    model.eval()
    device = model.device
    with torch.no_grad():
        for (labels, (text, text_len)), _ in test_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    y_pred = [np.argmax(y) for y in y_pred]
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=np.unique(y_true), digits=4))
