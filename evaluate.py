"""Evaluate saved CL-AMPNet fold ensembles on labeled test sequences."""
import argparse
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

import run_training as training
from src.common import (SCHEMA, check_sequence, digest, environment, feature_configs_match,
                        file_sha, read_csv, read_json, seed_everything, write_csv, write_json)

ROOT = Path(__file__).resolve().parent
DATASETS = ('APD3', 'dbAMP', 'DRAMP', 'LAMP', 'YADAMP')
METRICS = ('ACC', 'AUC', 'Sn', 'Sp', 'MCC', 'F1')


def read_tests(specs):
    tests = {}
    specs = specs or [f'{name}={ROOT / "data/main" / (name + "_test.csv")}' for name in DATASETS]
    for spec in specs:
        name, separator, filename = spec.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in tests:
            raise ValueError('Use unique test names: --test APD3=/path/to/APD3_test.csv')
        path = Path(filename).resolve()
        rows = read_csv(path)
        seen, labels = set(), set()
        for row in rows:
            sequence = row.get('sequence', '')
            check_sequence(sequence)
            if row.get('label') not in ('0', '1') or sequence in seen:
                raise ValueError(f'{name}: invalid label or duplicate sequence')
            seen.add(sequence)
            labels.add(int(row['label']))
        if labels != {0, 1}:
            raise ValueError(f'{name}: evaluation requires both AMP and non-AMP sequences')
        tests[name] = (path, rows)
    return tests


def verify_models(args, config, tests):
    models, reference = {}, None
    expected_features = training.expected_feature_config(config)
    for seed in args.seeds:
        run = args.runs / f'seed_{seed}'
        saved = read_json(run / 'config.json')
        done = read_json(run / 'TRAINING_COMPLETE.json')
        expected = dict(config['training'], schema=SCHEMA, seed=seed, variant='full',
                        contrastive_objective=training.OBJECTIVE)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{run}: settings differ from --config')
        config_sha = file_sha(run / 'config.json')
        if done.get('config_sha256') != config_sha:
            raise ValueError(f'{run}: training completion record does not match')
        snapshot = run / 'train_snapshot.csv'
        if file_sha(snapshot) != saved['train_sha256']:
            raise ValueError(f'{run}: training snapshot changed')
        feature_path = run / 'feature_config.json'
        if (file_sha(feature_path) != saved['feature_config_sha256']
                or not feature_configs_match(read_json(feature_path), expected_features)):
            raise ValueError(f'{run}: incompatible feature configuration')
        sources = saved['source_sha256']
        for name in ('model.py', 'dataset.py', 'features.py', 'train.py', 'metrics.py', 'common.py', 'losses.py'):
            if sources.get(name) != file_sha(ROOT / 'src' / name):
                raise ValueError(f'{run}: {name} differs from the training code; restore that version')
        identity = (saved['train_sha256'], file_sha(run / 'fold_assignments.csv'))
        if reference is not None and identity != reference:
            raise ValueError('Training data or fold assignments differ across seeds')
        reference = identity
        train_sequences = {row['sequence'] for row in read_csv(snapshot)}
        for name, (_, rows) in tests.items():
            overlap = sum(row['sequence'] in train_sequences for row in rows)
            if overlap:
                raise ValueError(f'{name}: {overlap} exact training matches; supply a separate test set')
        hashes = {}
        for fold in range(1, saved['folds'] + 1):
            path = run / f'fold_{fold}/best.pt'
            hashes[str(fold)] = file_sha(path)
            if hashes[str(fold)] != done['checkpoint_sha256'].get(str(fold)):
                raise ValueError(f'Checkpoint changed: {path}')
        models[seed] = dict(run=run, config=saved, config_sha256=config_sha, checkpoints=hashes)
    return models


def prepare_features(args, config, tests, rows):
    _, pending = training.check_cache(args.cache, rows, config)
    if not pending:
        return
    if args.stage == 'test':
        raise FileNotFoundError(f'{pending} test features missing; run --stage features or --stage all')
    features = config['features']
    if environment()['transformers'] != features['transformers']:
        raise RuntimeError('Feature extraction requires the Transformers version in requirements.txt')
    if features['t5_dtype'] == 'float16' and not args.device.startswith('cuda'):
        raise ValueError('The configured float16 feature extraction requires CUDA')
    command = [sys.executable, '-u', '-m', 'src.features', '--csv',
               *[str(path) for path, _ in tests.values()], '--cache', str(args.cache),
               '--device', args.device, '--t5-revision', features['t5_revision'],
               '--esm-revision', features['esm_revision']]
    if features['t5_dtype'] == 'float16':
        command.append('--t5-fp16')
    if args.hf_cache:
        command += ['--hf-cache', str(args.hf_cache)]
    if args.local_files_only:
        command.append('--local-files-only')
    subprocess.run(command, cwd=ROOT, check=True)
    if training.check_cache(args.cache, rows, config)[1]:
        raise RuntimeError('Test feature extraction is incomplete')


def evaluate(args, config, tests, rows, models):
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from src.dataset import AMPGraphDataset
    from src.metrics import binary_metrics
    from src.model import CLAMPNet
    from src.train import infer

    protocol = dict(model='CL-AMPNet', objective=training.OBJECTIVE, seeds=args.seeds,
                    threshold=args.threshold, device=args.device, batch_size=args.batch_size,
                    tests={name: file_sha(path) for name, (path, _) in tests.items()},
                    models={str(seed): dict(config_sha256=m['config_sha256'],
                                           checkpoints=m['checkpoints']) for seed, m in models.items()},
                    feature_config_sha256=file_sha(args.cache / 'feature_config.json'),
                    feature_records_sha256=digest(''.join(file_sha(args.cache / 'records' / f'{digest(row["sequence"])}.json')
                                                        for row in rows)),
                    source_sha256=file_sha(__file__), environment=environment())
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume:
            raise FileExistsError('Output directory is not empty; use --resume or a new --out directory')
        if read_json(args.out / 'evaluation_config.json') != protocol:
            raise ValueError('Evaluation settings or inputs changed; use a new --out directory')
    write_json(args.out / 'evaluation_config.json', protocol)
    evaluation_sha = file_sha(args.out / 'evaluation_config.json')
    results = []
    for name, (path, expected_rows) in tests.items():
        data = AMPGraphDataset(path, args.cache, config['training']['contact_threshold'])
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for seed, info in models.items():
            seed_everything(seed)
            out = args.out / name / f'seed_{seed}'
            prediction_path = out / 'predictions.csv'
            metric_path = out / 'metrics.json'
            if args.resume and metric_path.exists():
                saved = read_json(metric_path)
                if (saved.get('dataset') != name or saved.get('seed') != seed
                        or saved.get('evaluation_config_sha256') != evaluation_sha):
                    raise ValueError(f'Saved result belongs to a different evaluation: {metric_path}')
                if saved.get('predictions_sha256') != file_sha(prediction_path):
                    raise ValueError(f'Saved predictions changed: {prediction_path}')
                predictions = read_csv(prediction_path)
                if [(r['sequence'], r['label']) for r in predictions] != [(r['sequence'], r['label']) for r in expected_rows]:
                    raise ValueError(f'Saved test rows changed: {name}, seed {seed}')
                matrix = np.array([[float(r[f'fold_{k}_prob']) for k in range(1, config['training']['folds'] + 1)]
                                   for r in predictions])
                probability = np.array([float(r['probability']) for r in predictions])
                if (not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any()
                        or not np.allclose(matrix.mean(axis=1), probability, rtol=0, atol=1e-10)
                        or [int(r['prediction']) for r in predictions] != (probability >= args.threshold).astype(int).tolist()):
                    raise ValueError(f'Invalid saved ensemble: {name}, seed {seed}')
                metrics = binary_metrics(data.labels, probability, args.threshold)
                if any(saved.get(k) != v for k, v in metrics.items()):
                    raise ValueError(f'Saved metrics disagree: {name}, seed {seed}')
            else:
                probabilities = []
                for fold in range(1, config['training']['folds'] + 1):
                    model = CLAMPNet().to(args.device)
                    checkpoint = torch.load(info['run'] / f'fold_{fold}/best.pt', map_location=args.device, weights_only=True)
                    if (checkpoint.get('config_sha256') != info['config_sha256']
                            or checkpoint.get('fold') != fold or checkpoint.get('variant') != 'full'):
                        raise ValueError(f'Wrong checkpoint identity: seed {seed}, fold {fold}')
                    model.load_state_dict(checkpoint['state_dict'], strict=True)
                    labels, values = infer(model, loader, args.device)
                    if labels != data.labels:
                        raise ValueError('Inference label order differs from input')
                    probabilities.append(values)
                    del model, checkpoint
                matrix = np.asarray(probabilities, dtype=float).T
                probability = matrix.mean(axis=1)
                metrics = binary_metrics(data.labels, probability, args.threshold)
                predictions = [dict(id=row.get('id') or sid, sequence=row['sequence'], label=label,
                                    probability=float(p), prediction=int(p >= args.threshold),
                                    **{f'fold_{k}_prob': float(matrix[i, k-1]) for k in range(1, matrix.shape[1]+1)})
                               for i, (sid, row, label, p) in enumerate(zip(data.ids, data.rows, data.labels, probability))]
                write_csv(prediction_path, predictions)
                write_json(metric_path, dict(dataset=name, seed=seed, **metrics,
                           evaluation_config_sha256=evaluation_sha,
                           predictions_sha256=file_sha(prediction_path)))
            results.append(dict(dataset=name, seed=seed, **metrics))
            print(f'{name} seed={seed}: ACC={metrics["ACC"]:.4f} AUC={metrics["AUC"]:.4f} MCC={metrics["MCC"]:.4f}', flush=True)
    write_csv(args.out / 'per_run.csv', results)
    summary, lines = [], ['# CL-AMPNet test results', '',
        f'Threshold: {args.threshold:.10g}. Fold probabilities are averaged within each seed.',
        'Mean and sample SD are computed across seeds; a single run has no SD.', '',
        '|Dataset|Runs|N per run|ACC|AUC|Sn|Sp|MCC|F1|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in tests:
        group = [r for r in results if r['dataset'] == name]
        record = dict(dataset=name, runs=len(group), N_per_run=group[0]['N'], threshold=args.threshold)
        cells = []
        for metric in METRICS:
            values = [r[metric] for r in group]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            record[f'{metric}_mean'], record[f'{metric}_sd'] = mean, sd
            cells.append(f'{mean:.4f} ± {sd:.4f}' if sd is not None else f'{mean:.4f} (single run)')
        summary.append(record)
        lines.append('| ' + ' | '.join([name, str(len(group)), str(group[0]['N']), *cells]) + ' |')
    write_csv(args.out / 'summary.csv', summary)
    report = '\n'.join(lines) + '\n'
    (args.out / 'paper_tables.md').write_text(report, encoding='utf-8')
    print(report, flush=True)
    print(f'Results: {args.out}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'features', 'test', 'all'), default='check')
    parser.add_argument('--config', type=Path, default=ROOT / 'training_config.json')
    parser.add_argument('--test', nargs='+', metavar='NAME=CSV')
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/main')
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/features')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/main')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--hf-cache', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1 or args.batch_size < 1:
        parser.error('Use threshold in [0,1] and batch-size >= 1')
    for key in ('config', 'runs', 'cache', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    if args.hf_cache:
        args.hf_cache = args.hf_cache.resolve()
    config = training.configuration(args.config)
    args.seeds = config['seeds'] if args.seeds is None else args.seeds
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(type(s) is not int or not 0 <= s <= 2147483 for s in args.seeds)):
        parser.error('Use distinct nonnegative integer seeds from the completed training runs')
    tests = read_tests(args.test)
    models = verify_models(args, config, tests)
    unique = {row['sequence']: row for _, rows in tests.values() for row in rows}
    rows = [unique[s] for s in sorted(unique)]
    complete, pending = training.check_cache(args.cache, rows, config)
    for name, (_, subset) in tests.items():
        positives = sum(int(r['label']) for r in subset)
        print(f'{name}: N={len(subset)}, AMP={positives}, non-AMP={len(subset)-positives}; exact training overlap=0', flush=True)
    print(f'Models: {len(models)} seeds, {config["training"]["folds"]} folds each. Features: {complete} cached, {pending} pending.', flush=True)
    if args.stage == 'check':
        print('Checks passed. No inference, training, or model download performed.', flush=True)
        return
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate the training environment or use --device cpu')
    prepare_features(args, config, tests, rows)
    if args.stage == 'features':
        print('Test features ready. Next: python evaluate.py --stage test', flush=True)
        return
    evaluate(args, config, tests, rows, models)


if __name__ == '__main__':
    main()
