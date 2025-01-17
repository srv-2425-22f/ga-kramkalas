import matplotlib.pyplot as plt
import torch
from IPython import display

# plt.ion

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

def plot(train_loss, test_loss, train_acc, test_acc, train_time):
    plt.figure(figsize=(10, 5))  # Create a new figure
    plt.plot(train_loss, label="Train Loss", color="blue")
    plt.plot(test_loss, label="Test Loss", color="orange")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.title(f"Loss Over Epochs | Time to train: {train_time:.2f} s")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))  # Create another figure
    plt.plot(train_acc, label="Train Accuracy", color="blue", linestyle="--")
    plt.plot(test_acc, label="Test Accuracy", color="orange", linestyle="--")
    plt.xlabel(f"Number of Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")
    plt.title(f"Accuracy Over Epochs | Time to train: {train_time:.2f} s")
    plt.show()

def plot_live(train_loss, test_loss, train_acc, test_acc):
    display.clear_output(wait=True)
    display.display(plt.gcf())

    epochs = range(1, len(train_loss) + 1)
    
    #Plot losses
    plt.figure(1, figsize=(10, 5))  # Create a new figure
    plt.clf()
    plt.plot(epochs, train_loss, label="Train Loss", color="blue")
    plt.plot(epochs, test_loss, label="Test Loss", color="orange")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.title(f"Loss Over Epochs")

    #Plot accuracy
    plt.figure(2, figsize=(10, 5))  # Create another figure
    plt.clf()
    plt.plot(epochs, train_acc, label="Train Accuracy", color="blue", linestyle="--")
    plt.plot(epochs, test_acc, label="Test Accuracy", color="orange", linestyle="--")
    plt.xlabel(f"Number of Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")
    plt.title(f"Accuracy Over Epochs")

    plt.pause(0.1)