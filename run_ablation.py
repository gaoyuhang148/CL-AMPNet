"""Train and evaluate matched CL-AMPNet ablations."""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import run_training as core
from src.common import digest, environment, file_sha, read_csv, read_json, write_csv
from clamp_ablation import evaluate as evaluation
from clamp_ablation.model import VARIANTS

ROOT = Path(__file__).resolve().parent


def variant_args(args, variant):
    values = vars(args).copy()
    values['runs'] = args.full_runs if variant == 'full' and not args.train_full else args.runs / variant
    values['out'] = args.out / variant
    return SimpleNamespace(**values)


def check_runs(args, config, rows, tests, variant):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    selected = variant_args(args, variant)
    models = evaluation.verify_models(selected, config, tests, variant)
    labels = [int(r['label']) for r in rows]
    folds = np.zeros(len(rows), dtype=int)
    splitter = StratifiedKFold(n_splits=config['training']['folds'], shuffle=True,
                             random_state=config['training']['split_seed'])
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(rows)), labels), 1):
        folds[indices] = fold
    expected = [(digest(r['sequence']), str(r['label']), str(f)) for r, f in zip(rows, folds)]
    runtime = environment()
    for info in models.values():
        saved = info['config']
        if saved['train_sha256'] != file_sha(args.train):
            raise ValueError('Training data differ from the comparison model')
        if [(r['id'], r['label'], r['validation_fold']) for r in read_csv(info['run'] / 'fold_assignments.csv')] != expected:
            raise ValueError('Fold assignments differ from the comparison model')
        for key in ('python', 'numpy', 'torch', 'torch-geometric', 'scikit-learn'):
            if saved['environment'].get(key) != runtime.get(key):
                raise ValueError(f'Training runtime differs for {key}; activate the original environment')
        if args.stage in ('all', 'train') and saved['device_type'] != args.device.split(':')[0]:
            raise ValueError('Use the same training device type as the comparison model')
    return models


def summarize(args, config, tests):
    per_run, summary = [], []
    lines = ['# CL-AMPNet ablation results', '',
             f'Threshold: {args.threshold:.10g}. Fold probabilities are averaged within each seed.',
             'Values are mean and sample SD across seeds; a single run has no SD.', '',
             '|Dataset|Variant|Runs|N per run|ACC|AUC|Sn|Sp|MCC|F1|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in tests:
        for variant in args.variants:
            selected = variant_args(args, variant)
            per_run.extend(r for r in read_csv(selected.out / 'per_run.csv') if r['dataset'] == name)
            rec = next(r for r in read_csv(selected.out / 'summary.csv') if r['dataset'] == name)
            summary.append(rec)
            cells = []
            for metric in evaluation.METRICS:
                mean, sd = float(rec[f'{metric}_mean']), rec[f'{metric}_sd']
                cells.append(f'{mean:.4f} ± {float(sd):.4f}' if sd else f'{mean:.4f} (single run)')
            lines.append('| ' + ' | '.join([name, variant, rec['runs'], rec['N_per_run'], *cells]) + ' |')
    lines += ['', f'Full: CNN + GAT + SupCon (weight={config["training"]["alpha"]}).',
              'no_contrastive: CNN + GAT + CE; sequence_only: CNN + CE; graph_only: GAT + CE.',
              'Graph-only retains ProtT5 node features and ESM-2 contact edges.',
              'The same supplied threshold is applied to every variant and seed; no threshold search is performed.']
    write_csv(args.out / 'per_run.csv', per_run)
    write_csv(args.out / 'summary.csv', summary)
    report = '\n'.join(lines) + '\n'
    (args.out / 'paper_tables.md').write_text(report, encoding='utf-8')
    print(report, flush=True)
    print(f'DONE: {args.out / "paper_tables.md"}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'features', 'train', 'test', 'all'), default='check')
    p.add_argument('--config', type=Path, default=ROOT / 'training_config.json')
    p.add_argument('--train', type=Path, default=ROOT / 'data/xu/train.csv')
    p.add_argument('--test', nargs='+', metavar='NAME=CSV', default=[f'XUAMP={ROOT / "data/xu/XUAMP_test.csv"}'])
    p.add_argument('--cache', type=Path, default=ROOT / 'data/features')
    p.add_argument('--runs', type=Path, default=ROOT / 'runs/xu_ablation')
    p.add_argument('--full-runs', type=Path, default=ROOT / 'runs/xu')
    p.add_argument('--out', type=Path, default=ROOT / 'results/xu_ablation')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=32, help='Inference batch size; training uses training_config.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--train-full', action='store_true', help='Train full in --runs/full instead of reusing --full-runs')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--hf-cache', type=Path)
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args(argv)
    for key in ('config', 'train', 'cache', 'runs', 'full_runs', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    if args.hf_cache:
        args.hf_cache = args.hf_cache.resolve()
    config = core.configuration(args.config)
    args.seeds = config['seeds'] if args.seeds is None else args.seeds
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(type(s) is not int or not 0 <= s <= 2147483 for s in args.seeds)):
        p.error('Use distinct integer seeds in [0,2147483]')
    if len(set(args.variants)) != len(args.variants):
        p.error('Variants must be distinct')
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1 or args.batch_size < 1:
        p.error('Use threshold in [0,1] and batch-size >= 1')
    if args.train_full and args.runs / 'full' == args.full_runs:
        p.error('Use separate --runs and --full-runs directories')
    rows, counts = core.validate_training_data(args.train, config['training']['folds'], retain_duplicate_rows=True)
    tests = evaluation.read_tests(args.test)
    sequences = {r['sequence'] for r in rows}
    for name, (_, subset) in tests.items():
        overlap = sum(r['sequence'] in sequences for r in subset)
        if overlap:
            raise ValueError(f'{name}: {overlap} exact training matches')
        print(f'{name}: N={len(subset)}, AMP={sum(int(r["label"]) for r in subset)}, exact training overlap=0', flush=True)
    unique = {r['sequence']: r for _, subset in tests.values() for r in subset}
    test_rows = [unique[s] for s in sorted(unique)]
    ready, pending = core.check_cache(args.cache, rows + [r for r in test_rows if r['sequence'] not in sequences], config)
    print(f'Training: {len(rows)}, AMP={counts[1]}, non-AMP={counts[0]}. Features: {ready} cached, {pending} pending.', flush=True)
    print(f'Variants: {args.variants}; seeds: {args.seeds}; threshold: {args.threshold}', flush=True)
    if 'full' in args.variants and not args.train_full:
        check_runs(args, config, rows, tests, 'full')
        print(f'Reusing full models from {args.full_runs}', flush=True)
    if args.stage == 'check':
        print('Checks passed. No training, inference or download performed.', flush=True)
        return
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate the training environment or use --device cpu')
    if args.stage in ('features', 'all'):
        if core.check_cache(args.cache, rows, config)[1] and environment()['transformers'] != config['features']['transformers']:
            raise RuntimeError('Feature extraction requires the Transformers version in requirements.txt')
        core.extract_features(args, config, rows)
        evaluation.prepare_features(args, config, tests, test_rows)
        if args.stage == 'features':
            return
    if args.stage in ('train', 'all'):
        if core.check_cache(args.cache, rows, config)[1]:
            raise FileNotFoundError('Training features missing; run --stage features first')
        from clamp_ablation.train import train_run
        for variant in args.variants:
            if variant == 'full' and not args.train_full:
                continue
            selected = variant_args(args, variant)
            for seed in args.seeds:
                train_run(args.train, args.cache, selected.runs / f'seed_{seed}',
                          seed=seed, device=args.device, resume=args.resume, variant=variant, **config['training'])
    if args.stage in ('test', 'all'):
        evaluation.prepare_features(args, config, tests, test_rows)
        models = {v: check_runs(args, config, rows, tests, v) for v in args.variants}
        for variant in args.variants:
            evaluation.evaluate(variant_args(args, variant), config, tests, test_rows, models[variant], variant)
        summarize(args, config, tests)


if __name__ == '__main__':
    main()
