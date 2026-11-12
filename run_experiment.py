"""Run the primary or original-XU training and evaluation protocol."""
import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import run_training as training
import evaluate
from src.common import read_csv

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', choices=('main', 'xu'), required=True)
    p.add_argument('--stage', choices=('check', 'features', 'train', 'test', 'all'), default='check')
    p.add_argument('--config', type=Path, default=ROOT / 'training_config.json')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--cache', type=Path, default=ROOT / 'data/features')
    p.add_argument('--runs', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--hf-cache', type=Path)
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args(argv)
    config = training.configuration(a.config.resolve())
    seeds = a.seeds if a.seeds is not None else config['seeds']
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed <= 2147483 for seed in seeds)):
        p.error('Use distinct integer seeds in [0,2147483]')
    folder = ROOT / 'data' / a.protocol
    train = folder / 'train.csv'
    names = ('APD3', 'dbAMP', 'DRAMP', 'LAMP', 'YADAMP') if a.protocol == 'main' else ('XUAMP',)
    tests = evaluate.read_tests([f'{name}={folder / (name + "_test.csv")}' for name in names])
    rows, counts = training.validate_training_data(train, config['training']['folds'],
                                                   retain_duplicate_rows=a.protocol == 'xu')
    sequences = {r['sequence'] for r in rows}
    for name, (_, test_rows) in tests.items():
        overlap = sum(r['sequence'] in sequences for r in test_rows)
        if overlap:
            raise ValueError(f'{name}: {overlap} exact training matches')
        print(f'{name}: N={len(test_rows)}, AMP={sum(int(r["label"]) for r in test_rows)}, '
              'exact training overlap=0', flush=True)
    runs = (a.runs or ROOT / 'runs' / a.protocol).resolve()
    out = (a.out or ROOT / 'results' / a.protocol).resolve()
    for seed in seeds:
        snapshot = runs / f'seed_{seed}' / 'train_snapshot.csv'
        if snapshot.exists() and read_csv(snapshot) != rows:
            raise ValueError(f'{runs}: contains a different training dataset; use a new --runs directory')
    print(f'Protocol: {a.protocol}; training N={len(rows)} '
          f'(AMP={counts[1]}, Non-AMP={counts[0]}); seeds={seeds}', flush=True)
    print(f'Models: {runs}\nResults: {out}', flush=True)

    common = ['--config', str(a.config.resolve()), '--seeds', *map(str, seeds),
              '--cache', str(a.cache.resolve()), '--runs', str(runs), '--device', a.device]
    if a.resume:
        common += ['--resume']
    if a.hf_cache:
        common += ['--hf-cache', str(a.hf_cache.resolve())]
    if a.local_files_only:
        common += ['--local-files-only']
    train_args = ['--train', str(train)]
    if a.protocol == 'xu':
        train_args += ['--retain-duplicate-rows']
    if a.stage in ('check', 'features', 'train', 'all'):
        subprocess.run([sys.executable, '-u', str(ROOT / 'run_training.py'),
                        '--stage', a.stage, *train_args, *common], cwd=ROOT, check=True)
    if a.stage == 'check':
        test_rows = {r['sequence']: r for _, subset in tests.values() for r in subset}
        ready, pending = training.check_cache(a.cache.resolve(), list(test_rows.values()), config)
        print(f'Test features: {ready} cached, {pending} pending.', flush=True)
        print('Checks passed. Missing features will be extracted by --stage all.', flush=True)
    if a.stage == 'features':
        feature_args = SimpleNamespace(stage='features', cache=a.cache.resolve(), device=a.device,
                                       hf_cache=a.hf_cache, local_files_only=a.local_files_only)
        unique = {r['sequence']: r for _, subset in tests.values() for r in subset}
        evaluate.prepare_features(feature_args, config, tests, list(unique.values()))
        print('Training and test features ready.', flush=True)
    if a.stage in ('test', 'all'):
        subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate.py'), '--stage', a.stage,
                        '--test', *[f'{name}={path}' for name, (path, _) in tests.items()],
                        '--out', str(out), '--threshold', '0.5', *common], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
