import torch
import torch.nn.functional as F
from .attack import Attacker


class FGSM(Attacker):
    r"""
    'Explaining and Harnessing Adversarial Examples', https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model, epsilon: float = 6/255):
        super().__init__(model, epsilon=epsilon)

    def perturb(self, images: torch.Tensor, labels: torch.Tensor = None, targeted: bool = False):
        images_adv = images.clone().detach()

        images_adv = self.prepare_inputs(images_adv)

        loss = F.cross_entropy(self.forward(images_adv), labels)
        grad = torch.autograd.grad(loss, images_adv)[0]

        eta = self.epsilon * torch.sign(grad)

        if not targeted:
            images_adv = (images_adv + eta).detach()
        else:
            images_adv = (images_adv - eta).detach()

        images_adv = torch.clamp(images_adv, min=0, max=1.0)

        return self.unprepare_inputs(images_adv)

    def __repr__(self) -> str:
        return f'FGSM(eps={self.epsilon:>6.4f}({self.epsilon * 255.0:>.1f}/255.0), normalized={self.normalized}, mean={self.mean}, std={self.std})'
