import torch
from torch import nn

from typing import Dict, List, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect

def model_rgb2gray(model):
    # identify first layer

    first_layer = model
    while len(list(first_layer.children())) > 1:
        first_layer = list(first_layer.children())[0]

    # convert first layer to process grayscale image
    first_layer.in_channels = 1
    first_layer.weight = torch.nn.Parameter(first_layer.weight.sum(1, keepdim=True))


def create_figure(res: Dict[str, list], path: str, verbose: bool = True) -> None:
    """
    Create the figures accuracy, loss, mcc and ROC using the result of an experiment.

    :param res: a dictonary containing the result of the form {"epochs": [], "losses": [], "mcc": []...}
    :type res: Dict[str, list]
    :param path: the directory where save the figures
    :type path: str
    :param verbose: if True print messages during execution
    :type verbose: bool
    """
    epochs = res["epochs"]

    if verbose:
        print("Saving figures...")

    # loss
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_title(f"loss")
    ax.plot(epochs, res['losses'], label=f"train min={min(res['losses']):.4f} at epoch {np.argmin(res['losses']) + 1}")
    ax.plot(epochs, res['losses_val'], label=f"valid min={min(res['losses_val']):.4f} at epoch {np.argmin(res['losses_val']) + 1}")
    plt.yscale('log')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(path, "loss.png"))
    plt.close(fig)

    # train and validation accuracy
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_title(f"accuracy")
    ax.plot(epochs, res["train_accuracies"],
            label=f'train (train max={max(res["train_accuracies"]):.3f} at epoch {np.argmax(res["train_accuracies"]) + 1})')
    ax.plot(epochs, res["valid_accuracies"],
            label=f'valid (train max={max(res["valid_accuracies"]):.3f} at epoch {np.argmax(res["valid_accuracies"]) + 1})')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(path, "accuracy.png"))
    plt.close(fig)

    # mcc
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_title(f"mcc")
    ax.plot(epochs, np.abs(res["mcc_train"]),
            label=f'mcc_train (max={max(res["mcc_train"]):.3f} at epoch {np.argmax(res["mcc_train"]) + 1})')
    ax.plot(epochs, np.abs(res["mcc"]),
            label=f'mcc_valid (max={max(res["mcc"]):.3f} at epoch {np.argmax(res["mcc"]) + 1})')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(path, "mcc.png"))
    plt.close(fig)

    # roc at epoch with largest auc, only if res["auc"] is not empty (binary case)
    if len(res["auc"]) != 0:
        epoch_best = np.argmax(res["auc"])
        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        ax.set_title(f"ROC")
        ax.plot(res["roc"][epoch_best][0], res["roc"][epoch_best][1],
                label=f'AUC={res["auc"][epoch_best]:.3f}')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(path, "roc.png"))
        plt.close(fig)

    if verbose:
        print("Done.")


def has_parameter(cls, param_name):
    init_signature = inspect.signature(cls.__init__)
    return param_name in init_signature.parameters


def estimate_gpu_memory(model, input_size, batch_size, dtype=torch.float32):
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    # Input data
    input_memory = batch_size * torch.tensor(input_size).prod().item() * torch.tensor([], dtype=dtype).element_size()
    # Forward activations
    # This is more complex to estimate as it depends on the model architecture and input size
    # For simplicity, assume each layer's output is of similar size to the input
    num_layers = sum(1 for _ in model.children())
    activation_memory = num_layers * input_memory
    # Optimizer states (e.g., for Adam, we have additional states per parameter)
    optimizer_memory = 2 * param_memory  # for Adam, it's approximately 2x the parameter memory
    total_memory = param_memory + input_memory + activation_memory + optimizer_memory
    return total_memory