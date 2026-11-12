import csv
import hashlib
import json
import os
import platform
import random
from pathlib import Path

SCHEMA = 'clamp-paper-v3'
AA = frozenset('ACDEFGHIKLMNPQRSTVWY')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check_sequence(sequence, min_len=11, max_len=100):
    if not isinstance(sequence, str) or not min_len <= len(sequence) <= max_len:
        raise ValueError(f'Sequence length must be in [{min_len}, {max_len}]')
    if set(sequence) - AA:
        raise ValueError('Sequence contains nonstandard residues')


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    from importlib.metadata import version, PackageNotFoundError
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('torch', 'torch-geometric', 'transformers', 'numpy', 'scikit-learn', 'sentencepiece'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def source_hashes():
    root = Path(__file__).parent
    return {p.name: file_sha(p) for p in sorted(root.glob('*.py'))}


def feature_configs_match(actual, expected):
    actual, expected = dict(actual), dict(expected)
    hashes = (actual.pop('extractor_sha256', None), expected.pop('extractor_sha256', None))
    # The original extractor differs only in comments and validation messages.
    supported = {
        'b3fba4cf28d1b4e5cc1f5049d79275fa8301c8550188434106a35644f1cb19fa',
        file_sha(Path(__file__).with_name('features.py')),
    }
    return actual == expected and all(value in supported for value in hashes)
