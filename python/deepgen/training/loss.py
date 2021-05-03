import torch
import torch.nn as nn
import torch.autograd as autograd


class Wasserstein1(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, input, target):
        return (-1)*self.weight*torch.mean(input*target)


class GradPenalty(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, disc, real_samples, fake_samples):
        alpha = torch.rand(real_samples.shape[0], 1, 1, 1,
                           device=real_samples.device)
        lerp_samples = alpha*real_samples + (1-alpha)*fake_samples
        lerp_samples.requires_grad_(True)
        lerp_preds = disc(lerp_samples)
        lerp_grads = autograd.grad(
            lerp_preds, lerp_samples, torch.ones_like(lerp_preds),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        lerp_grads = lerp_grads.view(lerp_grads.shape[0], -1)
        return torch.mean(torch.pow(lerp_grads.norm(p=2, dim=1)-1, 2))*self.weight
