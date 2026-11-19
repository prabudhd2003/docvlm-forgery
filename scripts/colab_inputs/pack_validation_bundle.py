from pathlib import Path
import json,hashlib,zipfile
root=Path(__file__).resolve().parent
m=json.loads((root/'subset.json').read_text())
for remote,wanted in m['files'].items():
 p=root/'staging'/remote.lstrip('/')
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:
  raise SystemExit('Missing or changed input: '+remote)
target=root/'csev_colab_bundle.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for name in ['source.tar','csev_receipts_v1.json','subset.json','validate_gpu.py']:
  z.write(root/name,name)
 for remote in m['files']:
  relative='staging/'+remote.lstrip('/')
  z.write(root/relative,relative)
print('Verified',len(m['files']),'input files. Upload:',target)
print('Bundle size: %.1f MiB'%(target.stat().st_size/2**20))
