import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path('/Users/ivansamuel/Documents/Projects/telos')
REPO = 'Kazenowoko/telos'
REVISION = '43a0b67f8feac32bea4f48995016047471296353'
PREFIX = 'checkpoints/corosred/unified/100m_r50_modal_h100_seed42_cad1/'
MANIFEST = ROOT / 'research/h100_100m_r50_download_2026-10-18/verification.json'

info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
files = sorted((f for f in info.siblings if f.rfilename.startswith(PREFIX)), key=lambda f: f.rfilename)
expected = {f'checkpoint_step_{i}.pt' for i in range(2000, 25433, 2000)} | {
    'checkpoint_final.pt', 'config.json', 'runtime.json',
    'dataset_source.json', 'launch_command.json', 'train.log',
}
assert {f.rfilename[len(PREFIX):] for f in files} == expected, 'Unexpected remote file inventory'
manifest = {
    'repo_id': REPO, 'revision': info.sha, 'destination': str(ROOT / PREFIX),
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'downloading', 'verified_files': [], 'errors': [],
}
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Downloading {len(files)} files, {sum(f.size or 0 for f in files)/1024**3:.2f} GiB', flush=True)


def download_verify(f):
    path = Path(hf_hub_download(
        repo_id=REPO, filename=f.rfilename, revision=REVISION,
        repo_type='model', local_dir=str(ROOT),
    ))
    size = path.stat().st_size
    if size != f.size:
        raise RuntimeError('File size mismatch')
    lfs = f.lfs
    expected_sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    expected_blob = getattr(f, 'blob_id', None)
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f'blob {size}\0'.encode()) if not expected_sha else None
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha256.update(chunk)
            if git_sha1 is not None:
                git_sha1.update(chunk)
    digest = sha256.hexdigest()
    if expected_sha:
        if digest != expected_sha:
            raise RuntimeError('LFS SHA-256 mismatch')
        method = 'LFS SHA-256'
    else:
        if expected_blob is None or git_sha1.hexdigest() != expected_blob:
            raise RuntimeError('Git blob SHA-1 mismatch or absent remote hash')
        method = 'Git blob SHA-1'
    return {
        'remote_path': f.rfilename, 'local_path': str(path), 'bytes': size,
        'sha256': digest, 'remote_lfs_sha256': expected_sha,
        'remote_git_blob_id': expected_blob, 'verification': method,
    }

with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(download_verify, f): f.rfilename for f in files}
    for future in as_completed(futures):
        name = futures[future]
        try:
            result = future.result()
            manifest['verified_files'].append(result)
            print(f'[{len(manifest["verified_files"])}/{len(files)}] Verified {Path(name).name} ({result["bytes"]/1024**2:.1f} MiB)', flush=True)
        except Exception as exc:
            manifest['errors'].append({'remote_path': name, 'error_type': type(exc).__name__})
            print(f'FAILED {Path(name).name}: {type(exc).__name__}', flush=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')

manifest['status'] = 'complete' if len(manifest['verified_files']) == len(files) and not manifest['errors'] else 'incomplete'
manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
manifest['verified_files'].sort(key=lambda x: x['remote_path'])
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print(f'{manifest["status"].upper()}: {len(manifest["verified_files"])}/{len(files)} files hash-verified', flush=True)
if manifest['status'] != 'complete':
    raise SystemExit(1)
