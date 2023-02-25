import numpy as np
import math


def linear_scheduler(beta_start: float, beta_stop: float, timestamp: int):
    betas = np.linspace(beta_start, beta_stop, timestamp)
    alphas = np.array([(1-beta)**0.5 for beta in betas])
    return betas, alphas


def _alpha_bar(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


def cosine_scheduler(timestamp: int, max_beta=0.999):
    betas = []
    for i in range(timestamp):
        t1 = i / timestamp
        t2 = (i + 1) / timestamp
        betas.append(min(1 - _alpha_bar(t2) / _alpha_bar(t1), max_beta))
    betas = np.array(betas)
    alphas = np.array([(1-beta)**0.5 for beta in betas])
    return betas, alphas
