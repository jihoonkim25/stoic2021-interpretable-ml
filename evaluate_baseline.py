# Evaluate the performance of the baseline model

import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataloaders import get_resampling_dataloaders_proto, get_slice_dataloaders_proto
from baselines.mlp import MLP
from protopnet.metrics import torch_metrics
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torchvision import models
from numpy import extract, load
from collections import OrderedDict


def evaluate(dl_test: DataLoader) -> None:
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    model = models.vgg19(pretrained=False)
    model = model.to(device)
    model_path = os.path.join(os.path.join(os.path.abspath("models"), "1001"), "checkpoint_100.pth")
    loaded_model = torch.load(model_path)
    layers_vgg19 = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 512)),
        ('activation1', nn.ReLU()),
        ('dropout1', nn.Dropout()),
        ('fc2', nn.Linear(512, 256)),
        ('activation2', nn.ReLU()),
        ('dropout2', nn.Dropout()),
        ('fc3', nn.Linear(256, 128)),
        ('activation3', nn.ReLU()),
        ('dropout3', nn.Dropout()),
        ('fc4', nn.Linear(128, 2)),
        ('out', nn.Sigmoid())
    ])).to(device)
    model.classifier = layers_vgg19
    model.load_state_dict(loaded_model.get("model_state_dict"))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_acc = 0.0
        test_f1 = 0.0
        test_prec = 0.0
        test_recall = 0.0

        for x_batch, y_batch in dl_test:
            x_batch = x_batch.to(device)
            y_batch = y_batch.float().to(device)
            test_pred = model(x_batch.float())
            test_loss += loss_fn(test_pred,
                                 y_batch.type(torch.LongTensor).to(device))

            acc, f1, prec, recall = torch_metrics(
                torch.softmax(test_pred, dim=1), y_batch, device=device)
            test_acc += acc.item()
            test_f1 += f1.item()
            test_prec += prec.item()
            test_recall += recall.item()

        print("loss", test_loss.cpu().item() / len(dl_test))

        print("acc", test_acc / len(dl_test))

        print("f1", test_f1 / len(dl_test))

        print("prec", test_prec / len(dl_test))

        print("recall", test_recall / len(dl_test))
    print("----")

    print('done evaluating!!')
    return

def extract_scalars():
    scalars_path = os.path.join(os.path.join(
        os.path.abspath("models"), "11"), "scalars.npz")
    scalars = load(scalars_path)
    lst = scalars.files
    items = {}
    for item in lst:
        items[item] = scalars[item]
    return items

def plot_metrics(items):
    print(items.keys())
    train_losses = items['training_losses']
    plt.plot(train_losses, label='test_loss')
    plt.title('Train loss plot of baseline model per batch (100 epochs)')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def main():
    args = parse_command_line()

    dl_train, dl_valid, dl_test = get_slice_dataloaders_proto(args['data_dir'],
                                                            {'train': args['metadata_train'],
                                                             'val': args['metadata_valid'],
                                                             'test': args['metadata_test']},
                                                            args['batch_sz'])
    # train! 
    print("Evaluating Train:")
    evaluate(dl_train)
    print("Evaluating Validation:")
    evaluate(dl_valid)
    print("Evaluating Test:")
    evaluate(dl_test)

    items = extract_scalars()

    #plot_metrics(items)



    return


def parse_command_line():
    parser = argparse.ArgumentParser()
    # get arguments for metadata, data location
    parser.add_argument('--metadata_train', action="store", type=str,
                        default='.\\metadata\\metadata_train_trimmed.npz')
    parser.add_argument('--metadata_valid', action="store",
                        type=str, default='.\\metadata\\metadata_valid_trimmed.npz')
    parser.add_argument('--metadata_test', action="store", type=str,
                        default='.\\metadata\\metadata_test_trimmed.npz')
    parser.add_argument('--data_dir', action="store",
                        type=str, default=".\\preprocessed")

    # get arguments for training hyperparam
    # if batch_sz = 1 then there is 1 same per batch! (n=50 batches)
    parser.add_argument('--batch_sz', action="store", type=int, default=1)

    return vars(parser.parse_args())

if __name__ == "__main__":
    main()
