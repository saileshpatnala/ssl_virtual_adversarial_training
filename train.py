import os
import argparse
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from models.utils import get_device, dbpedia_preprocess_lstm, BIN_DIR
from models.lstm import LSTM, train_lstm
from models.nn import NeuralNet, train_nn
from models.vat_lstm import train_vat_lstm
from models.vat_nn import train_vat_nn

# To get better stacktrace when training on GPU
# https://stackoverflow.com/questions/51691563/cuda-runtime-error-59-device-side-assert-triggered
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# For debugging
torch.autograd.set_detect_anomaly(True)

device = get_device()


def init_argparse():
    """
    Function to initialize argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model to be trained')
    parser.add_argument('-s', '--split', help='split of data to use for VAT')
    return parser


def lstm():
    """
    Function to read data from csv file, preprocess data and
    run training for LSTM model
    """
    vocab_size, train_iter, val_iter, test_iter, _, _ = dbpedia_preprocess_lstm()
    model = LSTM(num_classes=3, vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_lstm(train_iter, val_iter, model, optimizer)


def nn():
    """
    Function to read preprocessed data from binary file and
    run training for NeuralNet model
    """
    X_all = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_all_x.pkl')).iloc[:, 0:300]
    y_all = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_all_y.pkl'))

    X_all = X_all.replace(np.nan, 0)
    y_all.transpose().values.astype('int')

    X_all_tensor = torch.from_numpy(X_all.values).float().to(device)
    y_all_tensor = torch.from_numpy(y_all.values.flatten().astype('int')).long().to(device)

    model = NeuralNet(input_size=X_all.shape[1], out_size=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_nn(X_all_tensor, y_all_tensor, model, optimizer)


def vat_nn(split=1):
    """
    Function to read preprocessed data from binary file and
    run training for VAT-NN model
    """
    X_all = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_all_x.pkl')).iloc[:, 0:300]
    y_all = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_all_y.pkl'))
    if split == 2:
        X_labeled = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_x_split2.pkl'))
        y_labeled = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_y_split2.pkl'))
    else:
        X_labeled = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_x_split1.pkl'))
        y_labeled = pd.read_pickle(os.path.join(BIN_DIR, 'dbpedia_train_y_split1.pkl'))

    X_all = X_all.replace(np.nan, 0)
    X_labeled = X_labeled.replace(np.nan, 0)
    y_all.transpose().values.astype('int')

    X_all_tensor = torch.from_numpy(X_all.values).float().to(device)
    y_all_tensor = torch.from_numpy(y_all.values.astype('int')).long().to(device)
    X_labeled_tensor = torch.from_numpy(X_labeled.values).float().to(device)
    y_labeled_tensor = torch.from_numpy(y_labeled.transpose().values[0]).to(device)
    dataset = torch.utils.data.TensorDataset(X_all_tensor, y_all_tensor)

    model = NeuralNet(input_size=X_all.shape[1], out_size=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_vat_nn(dataset, X_labeled_tensor, y_labeled_tensor, model, optimizer)


def vat_lstm(split=1):
    """
    Function to
    """
    vocab_size, train_iter, _, _, train_split_iter, _ = dbpedia_preprocess_lstm(split)
    model = LSTM(num_classes=3, vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_vat_lstm(train_iter, train_split_iter, model, optimizer)


def main():
    parser = init_argparse()
    args = parser.parse_args()
    if args.model == 'nn':
        print('Running NeuralNet Model')
        nn()
    elif args.model == 'lstm':
        print('Running LSTM Model')
        lstm()
    elif args.model == 'vat_nn':
        if args.split:
            print(f'Running VAT-NN Model on split {args.split}')
            vat_nn(args.split)
        else:
            print('Running VAT-NN Model on split 1')
            vat_nn()
    elif args.model == 'vat_lstm':
        if args.split:
            print(f'Running VAT-LSTM Model on split {args.split}')
            vat_lstm(args.split)
        else:
            print('Running VAT-LSTM Model on split 1')
            vat_lstm()
    else:
        print('Invalid model')


if __name__ == '__main__':
    main()
