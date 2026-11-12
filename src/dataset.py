from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .common import SCHEMA, check_sequence, digest, file_sha, read_csv, read_json
from .features import validate_arrays


def contact_edges(contacts, threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError('Contact threshold must be in (0,1)')
    if not np.allclose(contacts, contacts.T, atol=1e-5, rtol=0):
        raise ValueError('Asymmetric contacts')

    mask = ((contacts + contacts.T) * 0.5) >= threshold
    np.fill_diagonal(mask, False)
    return torch.from_numpy(np.stack(np.nonzero(mask))).long()


class AMPGraphDataset(Dataset):
    def __init__(self, csv, cache, contact_threshold=0.5, verify=True, unlabeled=False, allow_duplicate_rows=False):
        self.rows = read_csv(csv)
        self.cache = Path(cache)
        self.threshold = contact_threshold
        self.feature_config = read_json(self.cache / 'feature_config.json')
        if self.feature_config.get('schema') != SCHEMA or self.feature_config.get('algorithm') != 'prott5_residue_last_hidden__esm2_pretrained_contact_head_v1':
            raise ValueError('Unsupported feature cache')
        self.config_sha = file_sha(self.cache / 'feature_config.json')
        if not self.rows:
            raise ValueError('Empty dataset')
        self.ids, self.labels = [], []
        for row in self.rows:
            seq = row['sequence']
            check_sequence(seq)
            sid = digest(seq)
            if sid in self.ids and not allow_duplicate_rows:
                raise ValueError('Duplicate sequence in dataset')
            label = -1 if unlabeled else int(row['label'])
            if not unlabeled and label not in (0, 1):
                raise ValueError('Expected binary labels')
            self.ids.append(sid)
            self.labels.append(label)
            rec = read_json(self.cache / 'records' / f'{sid}.json')
            if rec['sequence'] != seq or rec['feature_config_sha256'] != self.config_sha:
                raise ValueError(f'Feature record mismatch for {sid}')
            ep, cp = self.paths(sid)
            if verify and (file_sha(ep) != rec['embedding_sha256'] or file_sha(cp) != rec['contacts_sha256']):
                raise ValueError(f'Feature checksum mismatch for {sid}')

    def paths(self, sid):
        return self.cache / 'embeddings' / f'{sid}.npy', self.cache / 'contacts' / f'{sid}.npy'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ep, cp = self.paths(self.ids[idx])
        emb, contacts = np.load(ep, allow_pickle=False), np.load(cp, allow_pickle=False)
        validate_arrays(self.rows[idx]['sequence'], emb, contacts)
        return Data(x=torch.from_numpy(emb.astype(np.float32)),
                    edge_index=contact_edges(contacts, self.threshold),
                    y=torch.tensor([self.labels[idx]], dtype=torch.long),
                    sample_id=self.ids[idx])
