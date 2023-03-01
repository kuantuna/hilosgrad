import math
import torch


class Diffusion:
    def __init__(self, num_steps=1000, noise_scheduler='linear') -> None:
        self.num_steps = num_steps
        self.noise_scheduler = noise_scheduler
        self.betas = None
        self.alphas = None
        self.alphas_hat = None
        self.set_scheduler = None
        self._prepare_noise_scheduler()

    def _prepare_noise_scheduler(self):
        valid_noise_schedulers = ['linear', 'cosine']
        if self.noise_scheduler not in valid_noise_schedulers:
            raise ValueError(
                f'{self.noise_scheduler} is not a valid scheduler.')
        if self.noise_scheduler == 'linear':
            def set_scheduler(beta_start: float = 0.0001, beta_stop: float = 0.02) -> None:
                self.betas = torch.linspace(
                    beta_start, beta_stop, self.num_steps)
                self.alphas = 1 - self.betas
                self.alphas_hat = torch.cumprod(self.alphas, dim=0)

        elif self.noise_scheduler == 'cosine':
            def alpha_bar(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            def set_scheduler(max_beta: float = 0.999) -> None:
                betas = []
                for i in range(self.num_steps):
                    t1 = i / self.num_steps
                    t2 = (i + 1) / self.num_steps
                    betas.append(
                        min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
                self.betas = torch.tensor(betas)
                self.alphas = 1 - self.betas
                self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.set_scheduler = set_scheduler

    def forward_process(self, image, t):
        alpha_hat = self.alphas_hat[t-1]
        squared_alpha_hat = torch.sqrt(alpha_hat)
        one_minus_squared_alpha_hat = torch.sqrt(1-alpha_hat)
        n_dist = torch.randn_like(image)
        return squared_alpha_hat*image + one_minus_squared_alpha_hat*n_dist

    def forward_process_step(self, image, T):
        for t in range(T):
            squared_alpha = torch.sqrt(self.alphas[t])
            squared_beta = torch.sqrt(self.betas[t])
            n_dist = torch.randn_like(image)
            image = squared_alpha*image + squared_beta*n_dist
        return image
