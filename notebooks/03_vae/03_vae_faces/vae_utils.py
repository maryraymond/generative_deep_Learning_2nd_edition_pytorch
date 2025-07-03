# This code is based on the code available for Generative deep learning - 2nd Edition book
# The original code is available [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition) and is licensed under the Apache License 2.0.
# This implementation is distributed under the Apache License 2.0. See the LICENSE file for details._

import numpy as np
import matplotlib.pyplot as plt
import torch


def get_vector_from_label(data_loader, vae, embedding_dim, device, type):
    curr_sum_pos = np.zeros(shape=embedding_dim, dtype=np.float32)
    curr_sum_neg = np.zeros(shape=embedding_dim, dtype=np.float32)

    curr_mean_pos = np.zeros(shape=embedding_dim, dtype=np.float32)
    curr_mean_neg = np.zeros(shape=embedding_dim, dtype=np.float32)

    curr_num_pos = 0
    curr_num_neg = 0

    curr_vector = np.zeros(shape=embedding_dim, dtype=np.float32)
    curr_dist = 0

    vector_found = False

    for batch in data_loader:
        # batch = next(data_iter)
        images = batch[0]
        attr = batch[1]

        # predict
        with torch.no_grad():
            vae.eval()
            _, _, z = vae.encoder.forward(images.to(device).to(type))
            z = z.to("cpu")

        z_pos = z[attr == 1]
        z_neg = z[attr == -1]

        if len(z_pos) > 0 :
            curr_sum_pos += torch.sum(z_pos, dim=0).numpy()
            curr_num_pos += len(z_pos)
            new_mean_pos = curr_sum_pos/curr_num_pos
            movement_pos = np.linalg.norm(new_mean_pos - curr_mean_pos)

        if len(z_neg) > 0 :
            curr_sum_neg += torch.sum(z_neg, dim=0).numpy()
            curr_num_neg += len(z_neg)
            new_mean_neg = curr_sum_neg/curr_num_neg
            movement_neg = np.linalg.norm(new_mean_neg - curr_mean_neg)

        curr_vector = new_mean_pos - new_mean_neg
        new_distance = np.linalg.norm(curr_vector)
        dist_change = new_distance - curr_dist
        print("distance change = ", dist_change)

        curr_mean_pos = np.copy(new_mean_pos)
        curr_mean_neg = np.copy(new_mean_neg)
        curr_dist = new_distance

        if np.sum([movement_neg, movement_pos]) < 0.08:
            curr_vector = curr_vector / curr_dist
            vector_found = True
            print("vector found")
            break
    
    if not vector_found:
        # get the best vector that we have anyway
        curr_vector = curr_vector / curr_dist
        
    return curr_vector

def add_vector_to_images(data_loader, vae, feature_vec, device, type):
    n_to_show = 5
    factors = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    data_iterator = iter(data_loader)
    example_batch = next(data_iterator)

    example_images = example_batch[0]

    vae.eval()

    with torch.no_grad():
        _, _, z_points = vae.encoder.forward(example_images.to(device).to(type))
        z_points = z_points.to("cpu").numpy()

    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):
        img = example_images[i].permute(1, 2, 0).numpy()
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis("off")
        sub.imshow(img)

        counter += 1

        for factor in factors:
            changed_z_point = z_points[i] + feature_vec * factor

            with torch.no_grad():
                changed_image = vae.decoder.forward(
                    torch.tensor([changed_z_point], device=device, dtype=type)
                )[0].to("cpu").permute(1, 2, 0).numpy()

            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis("off")
            sub.imshow(changed_image)

            counter += 1

    plt.show()


def morph_faces(data_loader, vae, device, type):
    factors = np.arange(0, 1, 0.1)

    data_iterator = iter(data_loader)
    example_batch = next(data_iterator)

    example_images = example_batch[0]
    vae.eval()

    with torch.no_grad():
        _, _, z_points = vae.encoder.forward(example_images.to(device).to(type))
        z_points = z_points.to("cpu").numpy()

    fig = plt.figure(figsize=(18, 8))

    counter = 1

    img = example_images[0].permute(1, 2, 0).numpy()
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    counter += 1

    for factor in factors:
        changed_z_point = z_points[0] * (1 - factor) + z_points[1] * factor

        with torch.no_grad():
            changed_image = vae.decoder.forward(
                torch.tensor([changed_z_point], device=device, dtype=type)
            )[0].cpu().permute(1, 2, 0).numpy()

        sub = fig.add_subplot(1, len(factors) + 2, counter)
        sub.axis("off")
        sub.imshow(changed_image)

        counter += 1

    img = example_images[1].permute(1, 2, 0).numpy()
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    plt.show()
