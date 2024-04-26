import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

def plot_decision_boundary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    """
    Plots decision boundaries of model predicting on x in comparison to y.
    Source :
      - https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py (with modifications)
      - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    x, y = x.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_0_min, x_0_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    x_1_min, x_1_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    x0x, x1x = np.meshgrid(np.linspace(x_0_min, x_0_max, 101), np.linspace(x_1_min, x_1_max, 101))

    # Make features
    x_to_pred_on = torch.from_numpy(np.column_stack((x0x.ravel(), x1x.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(x_to_pred_on)

    softmax = nn.Softmax(dim=1)
    y_log_proba = softmax(y_logits)
    y_pred = torch.argmax(y_log_proba, dim=1)

    # Reshape preds and plot
    y_pred = y_pred.reshape(x0x.shape).detach().numpy()
    plt.contourf(x0x, x1x, y_pred, alpha=0.6)
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.6, s=40)
    plt.xlim(x0x.min(), x0x.max())
    plt.ylim(x1x.min(), x1x.max())


def accuracy_fn(y_pred, y_true):
    """
    Calculates accuracy between predictions and truth labels.
    """
    try:
        if (len(y_pred) != len(y_true)):
            raise ValueError("Size Error")
    except ValueError:
        print("y_pred and y_true have not the same size !")
    else:
        size = len(y_pred)
        correct = torch.eq(y_pred, y_true).sum().item()
        acc = (correct / size) * 100
        return acc