import os
from pathlib import Path
import shutil

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Sampler, Subset
from torch_geometric.loader import DataLoader

from .common import SCHEMA, environment, file_sha, read_json, seed_everything, source_hashes, write_csv, write_json
from .dataset import AMPGraphDataset
from .metrics import binary_metrics
from .model import CLAMPNet
from .losses import joint_loss


class NonSingletonBatchSampler(Sampler):
    """Merge a final singleton into the preceding batch."""

    def __init__(self, size, batch_size, generator):
        if size < 2 or batch_size < 2:
            raise ValueError('Training requires at least two samples and batch_size >= 2')
        self.size, self.batch_size, self.generator = size, batch_size, generator

    def __iter__(self):
        indices = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [indices[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - int(self.size % self.batch_size == 1 and count > 1)


@torch.inference_mode()
def infer(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _, _ = model(batch)
        probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
        labels.extend(batch.y.cpu().tolist())
    return labels, probabilities


def atomic_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def train_run(train_csv, cache, run_dir, seed=42, split_seed=42,
              folds=5, epochs=15, batch_size=32, device='cuda', contact_threshold=0.5,
              lr=1e-4, weight_decay=1e-5, alpha=0.1, temperature=0.1,
              resume=False):
    train_csv, cache, run_dir = Path(train_csv), Path(cache), Path(run_dir)
    if folds < 2 or epochs < 1 or alpha <= 0 or temperature <= 0:
        raise ValueError('Invalid training configuration')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu explicitly')

    data = AMPGraphDataset(train_csv, cache, contact_threshold, allow_duplicate_rows=True)
    y = np.array(data.labels)
    if len(set(y)) != 2 or min(np.bincount(y)) < folds:
        raise ValueError('Not enough samples of each class for stratified folds')
    config = {'schema': SCHEMA, 'variant': 'full',
              'contrastive_objective': 'bidirectional_cross_view_label_multi_positive_mean_logprob_v1', 'seed': seed, 'split_seed': split_seed,
              'training_record_policy': 'retain_original_rows_and_labels', 'split_method': 'StratifiedKFold', 'folds': folds, 'epochs': epochs, 'batch_size': batch_size,
              'lr': lr, 'weight_decay': weight_decay, 'alpha': alpha, 'temperature': temperature,
              'threshold': 0.5, 'contact_threshold': contact_threshold, 'selection': 'max_validation_ACC_first_tie',
              'train_sha256': file_sha(train_csv),
              'feature_config_sha256': data.config_sha, 'source_sha256': source_hashes(),
              'environment': environment(), 'device_type': device.split(':')[0],
              'determinism': 'seeded; CUDA scatter reductions may remain nondeterministic',
              'singleton_policy': 'merge_into_previous_batch'}
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f'{run_dir} exists; use --resume or a new --runs directory')
        if read_json(run_dir / 'config.json') != config:
            raise ValueError('Run configuration mismatch; use a new --runs directory')
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir / 'config.json', config)
        write_json(run_dir / 'feature_config.json', data.feature_config)
        shutil.copyfile(train_csv, run_dir / 'train_snapshot.csv')
        (run_dir / 'source_snapshot').mkdir()
        for path in Path(__file__).parent.glob('*.py'):
            shutil.copyfile(path, run_dir / 'source_snapshot' / path.name)
    config_sha = file_sha(run_dir / 'config.json')
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    splits = list(splitter.split(np.zeros(len(y)), y))
    assignments = np.zeros(len(y), dtype=int)
    for fold, (_, val_indices) in enumerate(splits, 1):
        assignments[val_indices] = fold
    write_csv(run_dir / 'fold_assignments.csv', [{'id': sid, 'label': label, 'validation_fold': int(f)}
              for sid, label, f in zip(data.ids, data.labels, assignments)])
    oof = np.full(len(data), np.nan)
    summaries, checkpoints = [], {}
    for fold, (train_indices, val_indices) in enumerate(splits, 1):
        fold_dir = run_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        checkpoint = fold_dir / 'best.pt'
        completed = fold_dir / 'complete.json'
        seed_everything(seed * 1000 + fold)
        generator = torch.Generator().manual_seed(seed * 1000 + fold)
        train_set, val_set = Subset(data, train_indices.tolist()), Subset(data, val_indices.tolist())
        train_loader = DataLoader(train_set, batch_sampler=NonSingletonBatchSampler(len(train_set), batch_size, generator), num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        model = CLAMPNet().to(device)
        if completed.exists():
            done = read_json(completed)
            if done['config_sha256'] != config_sha or file_sha(checkpoint) != done['checkpoint_sha256']:
                raise ValueError(f'Completed fold {fold} has changed')
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            best_acc, best_epoch, history = -1.0, 0, []
            for epoch in range(1, epochs + 1):
                model.train()
                totals = np.zeros(3)
                count = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, z_s, z_g = model(batch)
                    loss, ce, cl = joint_loss(logits, batch.y, z_s, z_g, alpha, temperature)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss in fold {fold}, epoch {epoch}')
                    loss.backward()
                    optimizer.step()
                    totals += np.array([loss.item(), ce.item(), cl.item()]) * batch.num_graphs
                    count += batch.num_graphs
                truth, probabilities = infer(model, val_loader, device)
                metrics = binary_metrics(truth, probabilities)
                history.append({'epoch': epoch, 'loss': totals[0] / count, 'CE': totals[1] / count,
                                'CL': totals[2] / count, 'validation_ACC': metrics['ACC'], 'validation_AUC': metrics['AUC']})
                if metrics['ACC'] > best_acc:
                    best_acc, best_epoch = metrics['ACC'], epoch
                    atomic_checkpoint(checkpoint, {'state_dict': model.state_dict(), 'epoch': epoch,
                                      'config_sha256': config_sha, 'variant': 'full', 'fold': fold})
                write_csv(fold_dir / 'history.csv', history)
                print(f'full seed={seed} fold={fold}/{folds} epoch={epoch}/{epochs} loss={totals[0]/count:.5f} val_ACC={metrics["ACC"]:.5f}', flush=True)
            write_json(completed, {'config_sha256': config_sha, 'checkpoint_sha256': file_sha(checkpoint), 'best_epoch': best_epoch})
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved['config_sha256'] != config_sha or saved['fold'] != fold:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(saved['state_dict'])
        truth, probabilities = infer(model, val_loader, device)
        oof[val_indices] = probabilities
        summaries.append({'fold': fold, 'best_epoch': saved['epoch'], **binary_metrics(truth, probabilities)})
        checkpoints[str(fold)] = file_sha(checkpoint)
        write_csv(fold_dir / 'validation_predictions.csv', [{'id': data.ids[i], 'label': data.labels[i], 'probability': p}
                  for i, p in zip(val_indices, probabilities)])
        del model
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete OOF predictions')
    write_csv(run_dir / 'cv_metrics.csv', summaries)
    write_csv(run_dir / 'oof_predictions.csv', [{'id': sid, 'label': label, 'probability': float(p)} for sid, label, p in zip(data.ids, data.labels, oof)])
    write_json(run_dir / 'oof_metrics.json', {**binary_metrics(data.labels, oof),
               'note': 'OOF predictions from validation-selected checkpoints.'})
    write_json(run_dir / 'TRAINING_COMPLETE.json', {'config_sha256': config_sha, 'checkpoint_sha256': checkpoints})
    return run_dir
