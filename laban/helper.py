import matplotlib.pyplot as plt

# plt.ion

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