import math
from torch.nn import functional as F


def contrastive_loss(z_s, z_g, labels, temperature=0.1):
    """Bidirectional cross-view supervised contrastive loss."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if z_s.ndim != 2 or z_s.shape != z_g.shape or z_s.shape[0] < 2:
        raise ValueError('Expected matching [batch, features] tensors with batch >= 2')
    scores = F.normalize(z_s, dim=1) @ F.normalize(z_g, dim=1).T / temperature
    positives = labels[:, None].eq(labels[None, :]).to(scores.dtype)
    count = positives.sum(dim=1)
    sg = -(F.log_softmax(scores, dim=1) * positives).sum(dim=1) / count
    gs = -(F.log_softmax(scores.T, dim=1) * positives.T).sum(dim=1) / count
    return (sg.mean() + gs.mean()) / 2


def joint_loss(logits, labels, z_s, z_g, alpha=0.1, temperature=0.1):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Contrastive weight must be finite and nonnegative')
    ce = F.cross_entropy(logits, labels)
    cl = contrastive_loss(z_s, z_g, labels, temperature) if alpha > 0 else logits.new_zeros(())
    return ce + alpha * cl, ce, cl
