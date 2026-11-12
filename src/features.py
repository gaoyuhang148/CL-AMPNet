import argparse
import gc
import os
from pathlib import Path

import numpy as np

from .common import SCHEMA, check_sequence, digest, environment, file_sha, read_csv, read_json, write_json, feature_configs_match

T5_NAME = 'Rostlab/prot_t5_xl_uniref50'
ESM_NAME = 'facebook/esm2_t33_650M_UR50D'


def validate_arrays(sequence, embedding, contacts):
    check_sequence(sequence)
    n = len(sequence)
    if embedding.shape != (n, 1024) or contacts.shape != (n, n):
        raise ValueError(f'Feature length/dimension mismatch for {digest(sequence)}: {embedding.shape}, {contacts.shape}')
    if not np.isfinite(embedding).all() or not np.isfinite(contacts).all():
        raise ValueError('Nonfinite feature values')
    if (contacts < 0).any() or (contacts > 1).any():
        raise ValueError('Contact predictions must lie in [0,1]')
    if not np.allclose(contacts, contacts.T, atol=1e-5, rtol=0):
        raise ValueError('Contact head output is not symmetric')


def atomic_npy(path, array):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def token_residue_embeddings(output, special_mask, attention_mask, length):
    keep = attention_mask[0].bool() & ~special_mask[0].bool()
    result = output[0, keep]
    if result.shape != (length, 1024):
        raise ValueError(f'ProtT5 token alignment failed: {tuple(result.shape)} for {length} residues')
    return result


def main():
    p = argparse.ArgumentParser(description='Extract ProtT5 embeddings and ESM-2 contacts')
    p.add_argument('--csv', type=Path, nargs='+', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--hf-cache', type=Path)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--t5-revision', default='main')
    p.add_argument('--esm-revision', default='main')
    p.add_argument('--t5-fp16', action='store_true', help='Use float16 for ProtT5')
    a = p.parse_args()
    import torch
    from transformers import AutoConfig, T5EncoderModel, T5Tokenizer, EsmModel, EsmTokenizer
    from tqdm import tqdm
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check the environment or use --device cpu')
    if a.t5_fp16 and not a.device.startswith('cuda'):
        raise ValueError('--t5-fp16 requires CUDA')
    sequences = {}
    for csv_path in a.csv:
        for row in read_csv(csv_path):
            seq = row['sequence']
            check_sequence(seq)
            sequences[digest(seq)] = seq
    if not sequences:
        raise ValueError('No sequences')
    kwargs = {'cache_dir': str(a.hf_cache) if a.hf_cache else None,
              'local_files_only': a.local_files_only}

    t5_cfg = AutoConfig.from_pretrained(T5_NAME, revision=a.t5_revision, **kwargs)
    esm_cfg = AutoConfig.from_pretrained(ESM_NAME, revision=a.esm_revision, **kwargs)
    t5_rev, esm_rev = t5_cfg._commit_hash, esm_cfg._commit_hash
    if not t5_rev or not esm_rev:
        raise ValueError('Cannot resolve pretrained model revisions')
    config = {'schema': SCHEMA, 'algorithm': 'prott5_residue_last_hidden__esm2_pretrained_contact_head_v1',
              't5_model': T5_NAME, 't5_revision': t5_rev, 'esm_model': ESM_NAME, 'esm_revision': esm_rev,
              't5_dtype': 'float16' if a.t5_fp16 else 'float32', 'esm_dtype': 'float32',
              'transformers': environment()['transformers'], 'extractor_sha256': file_sha(__file__),
              'length_min': 11, 'length_max': 100, 'residue_policy': 'reject_nonstandard_no_truncation'}
    manifest = a.cache / 'feature_config.json'
    if manifest.exists():
        if not feature_configs_match(read_json(manifest), config):
            raise ValueError('Feature configuration mismatch; use a new --cache directory')
    else:
        if a.cache.exists() and any(a.cache.iterdir()):
            raise ValueError('Cache manifest missing; use a new --cache directory')
        a.cache.mkdir(parents=True, exist_ok=True)
        write_json(manifest, config)
    for name in ('embeddings', 'contacts', 'records'):
        (a.cache / name).mkdir(exist_ok=True)
    pending = []
    for sid, seq in sequences.items():
        rec_path = a.cache / 'records' / f'{sid}.json'
        ep, cp = a.cache / 'embeddings' / f'{sid}.npy', a.cache / 'contacts' / f'{sid}.npy'
        if rec_path.exists():
            rec = read_json(rec_path)
            if rec.get('sequence') != seq or rec.get('feature_config_sha256') != file_sha(manifest):
                raise ValueError(f'Cache record mismatch: {sid}')
            if file_sha(ep) != rec['embedding_sha256'] or file_sha(cp) != rec['contacts_sha256']:
                raise ValueError(f'Corrupted cache: {sid}')
            validate_arrays(seq, np.load(ep, allow_pickle=False), np.load(cp, allow_pickle=False))
        else:
            pending.append((sid, seq))
    if not pending:
        print(f'All {len(sequences)} sequence caches verified.')
        return

    tok = T5Tokenizer.from_pretrained(T5_NAME, revision=t5_rev, do_lower_case=False, **kwargs)
    model, info = T5EncoderModel.from_pretrained(T5_NAME, revision=t5_rev, output_loading_info=True,
                                               torch_dtype=torch.float16 if a.t5_fp16 else torch.float32, **kwargs)
    if info['missing_keys'] or info.get('mismatched_keys'):
        raise ValueError(f'Incomplete ProtT5 weights: {info}')
    model = model.to(a.device).eval().requires_grad_(False)
    for sid, seq in tqdm(pending, desc='ProtT5'):
        ep = a.cache / 'embeddings' / f'{sid}.npy'


        if ep.exists():
            arr = np.load(ep, allow_pickle=False)
            if arr.shape == (len(seq), 1024) and np.isfinite(arr).all():
                continue
            raise ValueError(f'Invalid partial embedding: {ep}')
        inputs = tok(' '.join(seq), add_special_tokens=True, return_special_tokens_mask=True, return_tensors='pt')
        special = inputs.pop('special_tokens_mask').to(a.device)
        inputs = {k: v.to(a.device) for k, v in inputs.items()}
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            emb = token_residue_embeddings(hidden, special, inputs['attention_mask'], len(seq))
        atomic_npy(ep, emb.float().cpu().numpy())
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tok = EsmTokenizer.from_pretrained(ESM_NAME, revision=esm_rev, **kwargs)
    model, info = EsmModel.from_pretrained(ESM_NAME, revision=esm_rev, add_pooling_layer=False,
                                         torch_dtype=torch.float32, output_loading_info=True, **kwargs)
    if info['missing_keys'] or info.get('mismatched_keys'):
        raise ValueError(f'Incomplete ESM-2 weights: {info}')
    model = model.to(a.device).eval().requires_grad_(False)
    for sid, seq in tqdm(pending, desc='ESM-2 contact head'):
        ep, cp = a.cache / 'embeddings' / f'{sid}.npy', a.cache / 'contacts' / f'{sid}.npy'
        if not cp.exists():
            inputs = tok(seq, add_special_tokens=True, return_tensors='pt')
            if inputs['input_ids'].shape[1] != len(seq) + 2:
                raise ValueError('ESM BOS/residue/EOS alignment failed')
            if inputs['input_ids'][0, 0].item() != tok.cls_token_id or inputs['input_ids'][0, -1].item() != tok.eos_token_id:
                raise ValueError('ESM contact head requires BOS/EOS tokens')
            inputs = {k: v.to(a.device) for k, v in inputs.items()}
            with torch.inference_mode():


                contacts = model.predict_contacts(inputs['input_ids'], inputs['attention_mask'])[0]
            atomic_npy(cp, contacts.float().cpu().numpy())
        validate_arrays(seq, np.load(ep, allow_pickle=False), np.load(cp, allow_pickle=False))
        write_json(a.cache / 'records' / f'{sid}.json',
                   {'sequence': seq, 'length': len(seq), 'feature_config_sha256': file_sha(manifest),
                    'embedding_sha256': file_sha(ep), 'contacts_sha256': file_sha(cp)})
    print(f'Features ready: {len(sequences)} unique sequences in {a.cache}')


if __name__ == '__main__':
    main()
