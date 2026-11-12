import argparse
from collections import Counter
import importlib.util
import math
from pathlib import Path
import re
import subprocess
import sys

from src.common import SCHEMA, check_sequence, digest, file_sha, read_csv, read_json, write_json, feature_configs_match

ROOT = Path(__file__).resolve().parent
TRAINING_KEYS = {'split_seed', 'folds', 'epochs', 'batch_size', 'contact_threshold',
                 'lr', 'weight_decay', 'alpha', 'temperature'}
OBJECTIVE = 'bidirectional_cross_view_label_multi_positive_mean_logprob_v1'


def configuration(path):
    config = read_json(path)
    if config.get('model') != 'CL-AMPNet' or config.get('objective') != OBJECTIVE:
        raise ValueError('Invalid model configuration')
    train, features = config['training'], config['features']
    if set(train) != TRAINING_KEYS:
        raise ValueError(f'Required training parameters: {sorted(TRAINING_KEYS)}')
    for key in ('split_seed', 'folds', 'epochs', 'batch_size'):
        if type(train[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if not 0 <= train['split_seed'] <= 2**32-1:
        raise ValueError('Invalid split seed')
    if train['folds'] < 2 or train['epochs'] < 1 or train['batch_size'] < 2:
        raise ValueError('Require folds >= 2, epochs >= 1, batch_size >= 2')
    for key in ('lr', 'weight_decay', 'alpha', 'temperature', 'contact_threshold'):
        if not isinstance(train[key], (int, float)) or not math.isfinite(train[key]):
            raise ValueError(f'Invalid parameter: {key}')
    if (min(train['lr'],train['alpha'],train['temperature']) <= 0 or train['weight_decay'] < 0
            or not 0 < train['contact_threshold'] < 1):
        raise ValueError('Invalid training parameters')
    if (features['t5_model'] != 'Rostlab/prot_t5_xl_uniref50'
            or features['esm_model'] != 'facebook/esm2_t33_650M_UR50D'
            or features['t5_dtype'] not in ('float16','float32') or features['esm_dtype'] != 'float32'):
        raise ValueError('Unsupported feature model or dtype')
    for key in ('t5_revision','esm_revision'):
        if not re.fullmatch('[0-9a-f]{40}',features[key]):
            raise ValueError('Model revisions must be 40-character commit hashes')
    return config


def validate_training_data(path, folds, retain_duplicate_rows=False):
    rows = read_csv(path)
    seen, counts = set(), Counter()
    for i,row in enumerate(rows,2):
        seq = row.get('sequence','')
        check_sequence(seq)
        if row.get('label') not in ('0','1'):
            raise ValueError(f'Invalid label in CSV line {i}; expected 0 or 1')
        if seq in seen and not retain_duplicate_rows:
            raise ValueError(f'Duplicate training sequence at CSV line {i}; use --retain-duplicate-rows for the original benchmark.')
        seen.add(seq)
        counts[int(row['label'])] += 1
    if set(counts) != {0,1} or min(counts.values()) < folds:
        raise ValueError('Too few samples per class for the requested folds')
    return rows,counts


def expected_feature_config(config):
    features = config['features']
    return {'schema':SCHEMA, 'algorithm':'prott5_residue_last_hidden__esm2_pretrained_contact_head_v1',
            't5_model':features['t5_model'],'t5_revision':features['t5_revision'],
            'esm_model':features['esm_model'],'esm_revision':features['esm_revision'],
            't5_dtype':features['t5_dtype'],'esm_dtype':features['esm_dtype'],
            'transformers':features['transformers'],'extractor_sha256':file_sha(ROOT/'src/features.py'),
            'length_min':11,'length_max':100,'residue_policy':'reject_nonstandard_no_truncation'}


def check_cache(cache, rows, config):
    manifest = cache/'feature_config.json'
    if not manifest.exists():
        if cache.exists() and any(cache.iterdir()):
            raise ValueError('Cache manifest missing; use a new --cache directory')
        return 0,len(rows)
    if not feature_configs_match(read_json(manifest), expected_feature_config(config)):
        raise ValueError(f'Feature configuration mismatch: {manifest}')
    config_sha, complete = file_sha(manifest),0
    for row in rows:
        sequence = row['sequence']; sid = digest(sequence)
        record_path = cache/'records'/f'{sid}.json'
        if not record_path.exists():
            continue
        rec = read_json(record_path)
        if rec.get('sequence') != sequence or rec.get('feature_config_sha256') != config_sha:
            raise ValueError(f'Feature record mismatch: {sid}')
        for folder,key in [('embeddings','embedding_sha256'),('contacts','contacts_sha256')]:
            path = cache/folder/f'{sid}.npy'
            if not path.is_file() or file_sha(path) != rec.get(key):
                raise ValueError(f'Feature file missing or corrupted: {path}')
        complete += 1
    return complete,len(rows)-complete


def extract_features(a, config, rows):
    complete, missing = check_cache(a.cache,rows,config)
    if not missing:
        print(f'Using {complete} cached features.', flush=True)
        return
    f = config['features']
    if f['t5_dtype'] == 'float16' and not a.device.startswith('cuda'):
        raise ValueError('ProtT5 float16 requires CUDA; use float32 with a separate cache for CPU.')
    command = [sys.executable,'-m','src.features','--csv',str(a.train),'--cache',str(a.cache),
               '--device',a.device,'--t5-revision',f['t5_revision'],'--esm-revision',f['esm_revision']]
    if f['t5_dtype'] == 'float16':
        command.append('--t5-fp16')
    if a.hf_cache:
        command += ['--hf-cache',str(a.hf_cache)]
    if a.local_files_only:
        command.append('--local-files-only')
    print(f'Extracting features for {missing} sequences.', flush=True)
    subprocess.run(command,cwd=ROOT,check=True)
    complete,missing = check_cache(a.cache,rows,config)
    if missing:
        raise RuntimeError('Feature extraction incomplete')


def train_models(a, config, rows):
    _,missing = check_cache(a.cache,rows,config)
    if missing:
        raise FileNotFoundError(f'{missing} features missing; run --stage features first')
    from src.train import train_run
    outputs = {}
    for seed in a.seeds:
        run = a.runs/f'seed_{seed}'
        train_run(a.train,a.cache,run,seed=seed,device=a.device,resume=a.resume,**config['training'])
        outputs[str(seed)] = {'directory':run.name,
                             'config_sha256':file_sha(run/'config.json'),
                             'completion_sha256':file_sha(run/'TRAINING_COMPLETE.json'),
                             'checkpoint_sha256':read_json(run/'TRAINING_COMPLETE.json')['checkpoint_sha256']}
    manifest = {'model':'CL-AMPNet','objective':OBJECTIVE,'seeds':a.seeds,
                'configuration_sha256':file_sha(a.config),'training_sha256':file_sha(a.train),
                'feature_config_sha256':file_sha(a.cache/'feature_config.json'),
                'entrypoint_sha256':file_sha(__file__),'runs':outputs}
    write_json(a.runs/('training_manifest_'+'_'.join(map(str,a.seeds))+'.json'),manifest)
    print(f'DONE: {a.runs}',flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description='CL-AMPNet training')

    # Input and output
    p.add_argument('--stage',choices=('check','features','train','all'),default='check')
    p.add_argument('--config',type=Path,default=ROOT/'training_config.json')
    p.add_argument('--train',type=Path,default=ROOT/'data/main/train.csv')
    p.add_argument('--cache',type=Path,default=ROOT/'data/features')
    p.add_argument('--runs',type=Path,default=ROOT/'runs/main')

    p.add_argument('--retain-duplicate-rows', action='store_true',
                   help='Retain all original benchmark rows and labels')

    # Training
    p.add_argument('--seeds',nargs='+',type=int,help='Default: the five seeds in training_config.json')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')

    # Pretrained weights
    p.add_argument('--hf-cache',type=Path)
    p.add_argument('--local-files-only',action='store_true')
    a = p.parse_args(argv)
    for key in ('config','train','cache','runs'):
        setattr(a,key,getattr(a,key).resolve())
    if a.hf_cache:
        a.hf_cache = a.hf_cache.resolve()
    config = configuration(a.config)
    a.seeds = config['seeds'] if a.seeds is None else a.seeds
    if (not a.seeds or len(set(a.seeds)) != len(a.seeds)
            or any(type(s) is not int or not 0 <= s <= 2147483 for s in a.seeds)):
        p.error('Use distinct integer seeds in [0,2147483]')
    rows,counts = validate_training_data(a.train,config['training']['folds'], a.retain_duplicate_rows)
    print(f'Training: {len(rows)} sequences; AMP={counts[1]}, non-AMP={counts[0]}.',flush=True)
    print(f'Full CL-AMPNet; SupCon weight={config["training"]["alpha"]}; '
          f'temperature={config["training"]["temperature"]}; seeds={a.seeds}.',flush=True)
    missing = [name for name in ('numpy','torch','torch_geometric','sklearn')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing packages: ' + ', '.join(missing))
    complete,pending = check_cache(a.cache,rows,config)
    print(f'Feature cache: {complete} verified, {pending} pending.',flush=True)
    if pending and a.stage in ('check','features','all'):
        missing = [name for name in ('transformers','sentencepiece')
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError('Missing feature-extraction packages: ' + ', '.join(missing))
        from src.common import environment
        if environment()['transformers'] != config['features']['transformers']:
            raise RuntimeError('Transformers version mismatch; install requirements.txt')
    if a.stage == 'check':
        print('Checks passed.',flush=True)
        return
    import torch
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check the environment or use --device cpu')
    if a.stage in ('features','all'):
        extract_features(a,config,rows)
    if a.stage in ('train','all'):
        train_models(a,config,rows)


if __name__ == '__main__':
    main()
