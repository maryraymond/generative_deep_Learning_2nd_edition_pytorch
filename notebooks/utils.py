# this code is modified from the Generative Deep Learning 2nd Edition repository 
# at (https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/utils.py)

# The original code is available [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition) and is licensed under the Apache License 2.0.
# This implementation is distributed under the Apache License 2.0. See the LICENSE file for details._

import matplotlib.pyplot as plt
import torch


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()
