import torch
from torch import nn
from torch.autograd import Variable
from torch import Tensor
import numpy as np

from dataloaders import get_resampling_dataloaders_proto, get_slice_dataloaders_proto
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


def saliency(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> np.ndarray: 
    """saliency

    Args:
        model (nn.Module): model for inference
        X (torch.Tensor): input tensor
        y (torch.Tensor): output tensor

    Returns:
        saliency map (np.ndarray)
    """
    model.eval()
    
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)

    prediction = model(X_var)

    prediction.backward(y_var.double())

    saliency = X_var.grad.data
    saliency = saliency.abs()
    saliency, i = torch.max(saliency,dim=1)
    saliency = saliency.squeeze() 

    #	print saliency.shape
    return saliency.data, prediction

def main():
    args = parse_command_line()

    dl_train, dl_valid, dl_test = get_slice_dataloaders_proto(args['data_dir'],
                                                              {'train': args['metadata_train'],
                                                               'val': args['metadata_valid'],
                                                               'test': args['metadata_test']},
                                                              args['batch_sz'])
    
    device = torch.device(
        'cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    model = models.vgg19(pretrained=False)
    model = model.to(device)
    model_path = os.path.join(os.path.join(
        os.path.abspath("models"), "1001"), "checkpoint_100.pth")
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

    for x_batch, y_batch in dl_test:
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.to(device)
        if (y_batch.cpu().item() == 1):
            y_true = torch.Tensor([[1, 0]])
        else: 
            y_true = torch.Tensor([[0, 1]])
        y_true = y_true.to(device)

        sal_data, pred = saliency(model, x_batch, y_true)
        saliency_map = Tensor.numpy(sal_data.cpu())
        print(y_true.cpu(), pred.cpu())
        plt.title("Saliency Map of Baseline VGG19 Model")
        #plt.title("true label: " + str(y_true.cpu()) + " predicted:", str(pred.cpu()), loc='right')
        plt.imshow(saliency_map, cmap='hot')
        plt.show()
    

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
