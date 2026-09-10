"""Real knowledge distillation loss (Hinton et al., 2015), as opposed to the
hard-label pseudo-labeling the rest of this repo uses. The difference: hard
labels keep only the teacher's single best answer and train the student with
ordinary cross-entropy against it; this keeps the teacher's full confidence
distribution over all three classes and trains the student to reproduce that
whole distribution, not just the winning class.

Why "temperature" matters: a teacher that's 99.98% sure of one answer (as
qwen3:14b often is -- see the worked examples in the README) gives the
student almost nothing to learn beyond "which class is highest," since the
other two classes are pinned near zero either way. Dividing by a temperature
T > 1 flattens the distribution before computing the loss, so the *relative*
confidence in the losing classes (which the argmax alone throws away) becomes
a real, learnable part of the target -- that relative ordering is exactly
the information hard labels can't carry.
"""
import torch
import torch.nn.functional as F


def soften(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Applies temperature T to an already-normalized probability
    distribution. Softmax(logits / T) is proportional to prob ** (1/T) when
    prob = softmax(logits) at T=1 -- so this reaches the same result as
    softening the teacher's underlying logits, without needing them."""
    softened = probs.clamp_min(1e-8).pow(1.0 / temperature)
    return softened / softened.sum(dim=-1, keepdim=True)


def distillation_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor,
                       temperature: float = 2.0) -> torch.Tensor:
    """Cross-entropy between the temperature-softened teacher distribution
    and the temperature-softened student distribution. The T^2 scaling
    follows Hinton et al. -- it keeps gradient magnitude comparable across
    different temperature choices."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = soften(teacher_probs, temperature)
    loss = -(teacher_soft * student_log_probs).sum(dim=-1).mean()
    return loss * (temperature ** 2)
