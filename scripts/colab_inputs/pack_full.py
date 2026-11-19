from pathlib import Path
import json,hashlib,zipfile
r=Path(__file__).resolve().parent;staging=r.parent/'csev_colab_validation/staging'
files=json.loads((r/'file_hashes.json').read_text())
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for i,(remote,h) in enumerate(files.items()):
 p=staging/remote.lstrip('/')
 if not p.is_file() or digest(p)!=h:raise SystemExit('Missing or changed: '+str(p))
 if (i+1)%100==0:print('Verified',i+1,'files',flush=True)
archive=r/'csev_full_inputs.zip';tmp=r/'csev_full_inputs.partial.zip'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
 for name in ['plan.json','csev_receipts_v1.json','source.tar','file_hashes.json','data_roles.json']:z.write(r/name,name)
 # Restore the standard TSV format config missing from the original test bundle.
 td=json.loads((r/'csev_receipts_v1.json').read_text())['legacy']['ocr']['tessdata_dir']
 z.writestr('staging/'+td.lstrip('/')+'/configs/tsv','tessedit_create_tsv 1\n')
 for remote in files:z.write(staging/remote.lstrip('/'),'staging/'+remote.lstrip('/'))
tmp.replace(archive)
print('Ready:',archive,'size GiB:',round(archive.stat().st_size/2**30,2))
