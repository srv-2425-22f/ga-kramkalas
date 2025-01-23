import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot_image_live(image):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(.1)