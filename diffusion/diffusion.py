import matplotlib.pyplot as plt
import numpy as np

from matplotlib import image
from schedulers import linear_scheduler

img = image.imread('cat.jpg')


# helper functions
def normalize(img):
    return img / 255 * 2 - 1


# def denormalize(img, data_type):
#     img = np.clip(img, -1.0, 1.0)
#     img = (img + 1) / 2 * 255
#     return np.rint(img).astype(data_type)

def denormalize(img):
    return (img + 1) / 2


# initialization
T = 1000
beta_start = 0.0001
beta_stop = 0.02
betas, alphas = linear_scheduler(beta_start, beta_stop, T)
print(f"{betas.shape=}")
print(f"{alphas.shape=}")


# forward diffusion process (step-by-step)
imgs = [normalize(img)]
for t in range(T):
    beta = betas[0]
    alpha = alphas[0]
    std = beta**0.5
    mean = alpha*imgs[t]
    normal_dist = np.random.randn(img.shape[0], img.shape[1], img.shape[2])
    noised_img = std*normal_dist + mean  # thanks to reparametrization trick
    imgs.append(noised_img)


# visualization
fig = plt.figure(figsize=(24, 24))
for idx, im in enumerate(imgs):
    if idx % 200 == 0:
        ax = fig.add_subplot(2, 3, int(idx/200)+1)
        ax.set_title(f"{idx}. iter")
        dim = denormalize(im)
        plt.imshow(dim)
plt.show()
