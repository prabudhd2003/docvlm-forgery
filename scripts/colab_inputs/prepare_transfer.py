from pathlib import Path
import json,shutil,hashlib
root=Path(__file__).resolve().parent
old=root.parent/'csev_colab_validation'
config=json.loads((old/'csev_receipts_v1.json').read_text());plan=json.loads((old/'plan.json').read_text());sub=json.loads((old/'subset.json').read_text())
assert hashlib.sha256((old/'plan.json').read_bytes()).hexdigest()==config['plan_sha256']
seen={d['record_id'] for d in sub['documents']}
files={}
for split in ['development','evaluation']:
 for d in plan[split]:
  for group in [d]+d['attacks']:
   for im in group['images']:
    assert im['path'].startswith('/')
    if im['path'] in files:assert files[im['path']]==im['sha256']
    files[im['path']]=im['sha256']
for lang,h in config['legacy']['ocr']['traineddata_sha256'].items():files[config['legacy']['ocr']['tessdata_dir']+'/'+lang+'.traineddata']=h
for name in ['plan.json','csev_receipts_v1.json','source.tar']:shutil.copy2(old/name,root/name)
(root/'remote_files.txt').write_text(''.join(p.lstrip('/')+'\n' for p in sorted(files)))
(root/'file_hashes.json').write_text(json.dumps(files,indent=2))
roles={'public_development':[d['record_id'] for d in plan['development']], 'collected_development':sorted(seen),'collected_reserved_evaluation':[d['record_id'] for d in plan['evaluation'] if d['record_id'] not in seen], 'notes':'Reserved evaluation has not been used in the present tuning session. Do not use its outcomes to choose prompts, crop geometry or thresholds. Existing historical analyses may have included these documents; this is not a claim of globally unseen data.'}
(root/'data_roles.json').write_text(json.dumps(roles,indent=2))
print('Input files:',len(files),'collected reserved:',len(roles['collected_reserved_evaluation']))
