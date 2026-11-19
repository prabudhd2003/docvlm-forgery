"""Fixed full-width band ablation; no answer-guided region selection."""
import argparse,json,sys,os,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--diagnostics',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);prior=Path(a.diagnostics);out=Path(a.out)
assert prior.is_dir();out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/'source/src'));os.environ['HF_HOME']=str(root/'hf_cache')
import torch
from PIL import Image,ImageDraw
from models.qwen import QwenVL
from csev.defense import Verifier,control_box,mask
from normalization import normalize_receipt_total

def band(box,size):
 if box is None:return None
 W,H=size;l,t,r,b=box
 # Fixed rule for every receipt: vertical context, full label-to-amount span.
 height=max(round(.12*H),b-t+2*round(.03*H))
 if height>.35*H:return None
 top=max(0,min(H-height,round((t+b-height)/2)))
 return (0,top,W,top+height)

c=json.loads((root/'csev_receipts_v1.json').read_text());subset=json.loads((root/'subset.json').read_text())
docs={d['record_id']:d for d in subset['documents']}
prior_rows=[json.loads(f.read_text()) for f in sorted(prior.glob('clean_*.json'))]
assert len(prior_rows)==15
(out/'protocol.json').write_text(json.dumps({'purpose':'Development-only band ablation','minimum_height_fraction':.12,'vertical_margin_fraction':.03,'maximum_height_fraction':.35,'threshold':0,'threshold_note':'Diagnostic only, not calibrated','localizers':['original_as_pixels','native_pixels'],'prior_directory':str(prior),'git_commit':c['git_commit']},indent=2))
w=QwenVL(device=torch.device('cuda'),**c['legacy']['model']);v=Verifier(w,c['settings'],c['legacy']['prompts']['receipt']);rows=[]
try:
 for i,old in enumerate(prior_rows):
  d=docs[old['record_id']];item=d['images'][0];path=root/'staging'/item['path'].lstrip('/')
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  im=Image.open(path).convert('RGB');ds=d['dataset_name'];prompt=c['legacy']['prompts']['receipt']
  full_raw=v.predict([im],prompt);full=normalize_receipt_total(full_raw,ds)
  row={'record_id':d['record_id'],'target':d['clean_target'],'full_raw':full_raw,'full':full,'variants':{}}
  for name in ['original_as_pixels','native_pixels']:
   box=band(old['variants'][name]['box'],im.size);r={'box':box,'prediction':None,'agrees':False,'gap':None,'accepted_at_zero':False}
   if box:
    crop=im.crop(box);raw=v.predict([crop],prompt);pred=normalize_receipt_total(raw,ds)
    r.update(raw=raw,prediction=pred,agrees=pred is not None and pred==full)
    crop.save(out/f'{i:02d}_{name}_band.png')
    overlay=im.copy();ImageDraw.Draw(overlay).rectangle(box,outline='red',width=max(3,im.width//200));overlay.thumbnail((800,1200));overlay.save(out/f'{i:02d}_{name}_overlay.png')
    control=control_box(box,im.size);r['control']=control
    if r['agrees'] and control:
     sc=v.score(mask(im,control,c['settings']['mask_value']),prompt,full)
     se=v.score(mask(im,box,c['settings']['mask_value']),prompt,full)
     # Also record actual answers after removing each region.
     raw_control=v.predict([mask(im,control,c['settings']['mask_value'])],prompt)
     raw_evidence=v.predict([mask(im,box,c['settings']['mask_value'])],prompt)
     r.update(score_control=sc,score_evidence=se,gap=sc-se,accepted_at_zero=sc-se>=0,masked_control_raw=raw_control,masked_evidence_raw=raw_evidence)
   row['variants'][name]=r
  (out/f'clean_{i:02d}.json').write_text(json.dumps(row,indent=2));rows.append(row)
  print(f'{i+1}/15 {d["record_id"]} full={full} target={d["clean_target"]} '+str({k:{'crop':r['prediction'],'gap':r['gap']} for k,r in row['variants'].items()}),flush=True)
 summary={'documents':len(rows),'full_correct':sum(r['full']==r['target'] for r in rows),'variants':{}}
 for name in ['original_as_pixels','native_pixels']:
  summary['variants'][name]={'agreement':sum(r['variants'][name]['agrees'] for r in rows),'correct_crops':sum(r['variants'][name]['prediction']==r['target'] for r in rows),'accepted_at_zero':sum(r['variants'][name]['accepted_at_zero'] for r in rows)}
 summary['limitations']='Clean development examples only; wider bands may weaken specificity. No claim of adversarial robustness.'
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
finally:w.unload()
