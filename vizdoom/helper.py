import matplotlib.pyplot as plt
from IPython import display
import torch
import numpy as np

plt.ion()

def plot_image_live(image, episode):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(f'Episode: {episode}')
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(.01)

def plot_loss_live(loss): 
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of episodes')
    plt.ylabel('Loss')
    
    plt.plot(loss)
    plt.text(len(loss)-1, loss[-1], str(loss[-1]))
    

    plt.show(block=False)
    plt.pause(.1)

def accuracy_fn(y_true, y_pred):
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

def plot(reward, episodes):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.figure(figsize=(10, 5))  # Create a new figure
    plt.ylim(top=100)
    # plt.ylim(bottom=0)
    # plt.yscale("log")
    plt.plot(reward, label="Loss", color="blue")    
    plt.xlabel("Number of Episodes")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.title(f"Loss Over Epochs | Episodes: {episodes}")
    plt.show(block=True)