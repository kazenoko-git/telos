"""Package an explicit, credential-free study source allowlist."""
import hashlib,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
manifest=json.loads((ROOT/'manifest.json').read_text());paths={REPO/p for p in manifest['sources']}
for directory in [REPO/'reports/xerj_reader_2026-11-14',REPO/'reports/specialist_extraction_2026-11-15',REPO/'reports/nuextract3_2026-11-15']:
    paths.update(directory.glob('*.py'));paths.update(directory.glob('*.md'))
paths.update([REPO/'LICENSE',REPO/'pyproject.toml',REPO/'reports/xerj_reader_2026-11-14/pipeline_task_plans.json',REPO/'reports/xerj_reader_2026-11-14/extended_cases.json',REPO/'reports/specialist_extraction_2026-11-15/gliner2_base_native_order.json'])
paths.update(ROOT/name for name in ['Reader-1-evaluation.pdf','report.html','results.json','manifest.json','package_audit.json','REPRODUCE.md','replay_saved_scores.py','build_report.py','package_bundle.py'])
# Keep excluded Qwen 2B outcomes out of the sharing bundle.
paths={p for p in paths if 'qwen2b' not in p.name.lower()}
checksums={}
for path in sorted(paths):
    raw=path.read_bytes()
    if path.suffix in ('.py','.json','.md','.html'):
        text=raw.decode('utf-8')
        if re.search(r'AIza[0-9A-Za-z_-]{30,}|xerj_[0-9A-Za-z]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text):raise ValueError('Credential-like data found in '+str(path.relative_to(REPO)))
    checksums[str(path.relative_to(REPO))]=hashlib.sha256(raw).hexdigest()
with zipfile.ZipFile(ROOT/'Reader-1-reproduction.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for path in sorted(paths):z.write(path,path.relative_to(REPO))
    z.write(ROOT/'Reader-1-evaluation.pdf','Reader-1-evaluation.pdf')
    z.writestr('START_HERE.md','Open Reader-1-evaluation.pdf for the three-page report. Full searchable report, source records, and reproduction instructions are under reports/reader1_consolidated_2026-11-15/. Run python3 reports/reader1_consolidated_2026-11-15/replay_saved_scores.py to replay all saved scores offline.\n')
    checksums['Reader-1-evaluation.pdf']=hashlib.sha256((ROOT/'Reader-1-evaluation.pdf').read_bytes()).hexdigest()
    z.writestr('SHA256SUMS.json',json.dumps(checksums,indent=2)+'\n')
print('Packaged',len(paths),'explicit files;', (ROOT/'Reader-1-reproduction.zip').stat().st_size,'bytes; credential-pattern scan passed')
