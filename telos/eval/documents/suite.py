"""Versioned document diagnostics shipped inside wheels and source distributions."""
import gzip
import hashlib
from importlib.resources import files
import json


def resource(name):
    data = files(__package__).joinpath('data', name).read_bytes()
    manifest = json.loads(files(__package__).joinpath('data', 'manifest.json').read_bytes())
    if name in manifest['files'] and hashlib.sha256(data).hexdigest() != manifest['files'][name]:
        raise ValueError(f'Packaged document suite resource hash mismatch: {name}')
    return data


def load_suite():
    cases = json.loads(resource('cases.json'))
    plans = json.loads(resource('plans.json'))['cases']
    by_id = {case['id']: case for case in cases}
    if len(by_id) != 512 or set(by_id) != {plan['case_id'] for plan in plans}:
        raise ValueError('Document suite must contain 512 unique cases and matching plans')
    return cases, {plan['case_id']: plan for plan in plans}, json.loads(resource('manifest.json'))


def reference_responses():
    return json.loads(gzip.decompress(resource('references.json.gz')))
