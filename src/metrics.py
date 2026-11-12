import math
import numpy as np


def binary_metrics(labels, probabilities, threshold=0.5):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Metrics require equal nonempty 1-D arrays')
    if not set(y.tolist()) <= {0, 1} or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid labels/probabilities')
    if not 0 <= threshold <= 1:
        raise ValueError('Invalid threshold')
    pred = p >= threshold
    tp, tn = int(((y == 1) & pred).sum()), int(((y == 0) & ~pred).sum())
    fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    pos, neg = tp + fn, tn + fp
    denom = math.sqrt((tp + fp) * pos * neg * (tn + fn))
    auc = None
    if pos and neg:
        order = np.argsort(p, kind='stable')
        sorted_p, sorted_y = p[order], y[order]
        rank_sum, i = 0.0, 0
        while i < len(y):
            j = i + 1
            while j < len(y) and sorted_p[j] == sorted_p[i]:
                j += 1
            rank_sum += (i + 1 + j) / 2 * int(sorted_y[i:j].sum())
            i = j
        auc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    return {'N': len(y), 'threshold': threshold, 'ACC': (tp + tn) / len(y),
            'AUC': auc, 'Sn': tp / pos if pos else None, 'Sp': tn / neg if neg else None,
            'MCC': (tp * tn - fp * fn) / denom if denom else 0.0,
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
