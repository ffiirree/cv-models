import torch
import torch.nn.functional as F
from .attack import Attacker


class PGD(Attacker):
    r"""
    'Towards Deep Learning Models Resistant to Adversarial Attacks', https://arxiv.org/abs/1706.06083
    """

    def __init__(self, model, epsilon: float = 0.05, steps: int = 7, alpha: float = 0.01):
        super().__init__(model, epsilon=epsilon)

        self.steps = steps
        self.alpha = alpha

    def perturb(self, images: torch.Tensor, labels: torch.Tensor = None, targeted: bool = False):
        images_adv = images.detach().clone()

        images_adv = self.prepare_inputs(images_adv)
        images_nat = images_adv.clone().detach()

        for _ in range(self.steps):
            images_adv.requires_grad_(True)

            loss = F.cross_entropy(self.forward(images_adv), labels)
            grad = torch.autograd.grad(loss, images_adv)[0]

            eta = self.alpha * torch.sign(grad)

            if not targeted:
                images_adv = (images_adv + eta).detach()
            else:
                images_adv = (images_adv - eta).detach()

            images_adv = torch.clamp(images_adv, images_nat - self.epsilon, images_nat + self.epsilon)
            images_adv = torch.clamp(images_adv, min=0, max=1.0)

        return self.unprepare_inputs(images_adv)

    def __repr__(self) -> str:
        return f'PGD(eps={self.epsilon}, steps={self.steps}, alpha={self.alpha}, normalized={self.normalized}, mean={self.mean}, std={self.std})'
