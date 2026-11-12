import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import unbatch

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(input_dim, 64, k, padding=k // 2) for k in (3, 5, 7))

        self.reduce = nn.Sequential(nn.BatchNorm1d(192), nn.GELU(), nn.Linear(192, 128))

    def forward(self, x, batch):
        pooled = []
        for seq in unbatch(x, batch):
            seq = seq.T.unsqueeze(0)
            pooled.append(torch.cat([F.relu(conv(seq)).amax(dim=-1) for conv in self.convs], dim=1))
        return self.reduce(torch.cat(pooled, dim=0))


class GraphEncoder(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()

        self.gat1 = GATConv(input_dim, 32, heads=4, concat=True, add_self_loops=True, dropout=0.0)
        self.gat2 = GATConv(128, 32, heads=4, concat=True, add_self_loops=True, dropout=0.0)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return global_mean_pool(x, batch)


class CLAMPNet(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.sequence = SequenceEncoder(input_dim)
        self.graph = GraphEncoder(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(128, 2))

    def forward(self, data):
        z_s = self.sequence(data.x, data.batch)
        z_g = self.graph(data.x, data.edge_index, data.batch)
        fused = torch.cat([z_s, z_g], dim=1)
        return self.classifier(fused), z_s, z_g
