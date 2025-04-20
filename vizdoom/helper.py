import matplotlib.pyplot as plt
from IPython import display
import torch
import numpy as np
import time

plt.ion()

def plot_image_live(image, episode):
    """Display the current game image in a live updating plot.
    
    Args:
        image (np.ndarray): The game screen image to display (shape: H x W x C)
        episode (int): Current episode number for title display
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(f'Episode {episode}')
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(.01)

def plot_loss_live(loss): 
    """Plot the training loss in a live updating graph.
    
    Args:
        loss (list): List of loss values to plot
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of episodes')
    plt.ylabel('Loss')    
    plt.plot(loss)
    plt.text(len(loss)-1, loss[-1], str(loss[-1]))
    plt.show(block=False)
    plt.pause(.01)

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def plot(episodes, loss, rewards, kd):
    """Create a comprehensive summary plot showing training metrics.
    
    Args:
        episodes (int): Total number of episodes completed
        loss (list): List of loss values over training
        rewards (list): List of reward values over training
        kd (list): List of kill/death ratio values over training
    """
    kd = [int(x) for x in kd]
    print(kd)
    print(rewards)

    # kd = [1,2,4,3,5]
    for i in range(len(rewards)):
        reward = np.float32(rewards[i])
        rewards[i] = reward
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
    axes[0].plot(loss, label="Loss", color="blue")
    axes[1].plot(rewards, label="Reward", color="red")
    axes[2].plot(kd, label="Kills", color="red")
    axes[0].set_title(f"Loss Over Epochs | Episodes: {episodes}")
    axes[1].set_title(f"Reward Over Epochs | Episodes: {episodes}")
    axes[2].set_title(f"Kills Over Epochs | Episodes: {episodes}")
    plt.legend(loc="best")
    plt.show(block=True)