import torch
from torch import nn

from src.model import SequenceEncoder, GraphEncoder

VARIANTS = ('full', 'no_contrastive', 'sequence_only', 'graph_only')


class AblationNet(nn.Module):
    def __init__(self, variant='full', input_dim=1024):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.variant = variant
        self.sequence = SequenceEncoder(input_dim) if variant != 'graph_only' else None
        self.graph = GraphEncoder(input_dim) if variant != 'sequence_only' else None
        width = 256 if variant in ('full', 'no_contrastive') else 128
        self.classifier = nn.Sequential(
            nn.Linear(width, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(128, 2))

    def forward(self, data):
        z_s = self.sequence(data.x, data.batch) if self.sequence is not None else None
        z_g = self.graph(data.x, data.edge_index, data.batch) if self.graph is not None else None
        fused = torch.cat([z_s, z_g], dim=1) if z_s is not None and z_g is not None else (z_s if z_s is not None else z_g)
        return self.classifier(fused), z_s, z_g
