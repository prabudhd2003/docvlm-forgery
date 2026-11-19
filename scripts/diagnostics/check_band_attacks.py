"""Frozen development attack check, with per-case Drive saves and resume."""
import argparse,json,sys,os,hashlib,math,time,io,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'source/src'));os.environ['HF_HOME']=str(root/'hf_cache')
import torch
from PIL import Image
from models.qwen import QwenVL
from csev.defense import Verifier,parse_box,control_box,mask
from normalization import normalize_receipt_total

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
def pixel_box(raw,size,rs,settings):
 try:
  value=json.loads(raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())['box']
  if not isinstance(value,list) or len(value)!=4:return None
  if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):return None
  return parse_box(json.dumps({'box':[x*1000/d for x,d in zip(value,(*rs,*rs))]}),size,settings)
 except (ValueError,TypeError,KeyError):return None

def band(box,size):
 if box is None:return None
 W,H=size;l,t,r,b=box;height=max(round(.12*H),b-t+2*round(.03*H))
 if height>.35*H:return None
 top=max(0,min(H-height,round((t+b-height)/2)))
 return (0,top,W,top+height)

c=json.loads((root/'csev_receipts_v1.json').read_text());s=json.loads((root/'subset.json').read_text())
protocol={'script_sha256':digest(__file__),'config_sha256':digest(root/'csev_receipts_v1.json'),'subset_sha256':digest(root/'subset.json'),'threshold':0,'band_min_height':.12,'vertical_margin':.03,'maximum_height':.35,'localizer':'original prompt, interpreted in processed-image pixels','limitations':'Development set only. Saved attacks were not optimized against this revised defense. Zero threshold not calibrated.'}
if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol,'Protocol changed: choose a new output directory.'
else:write(out/'protocol.json',protocol)
w=QwenVL(device=torch.device('cuda'),**c['legacy']['model']);v=Verifier(w,c['settings'],c['legacy']['prompts']['receipt']);prompt=v.prompt;patch=w.processor.image_processor.patch_size
assert isinstance(patch,int)

def observe(image,dataset):
 norm=lambda x:normalize_receipt_total(x,dataset)
 raw=v.predict([image],prompt);full=norm(raw)
 result={'full_raw':raw,'full_prediction':full,'crop_prediction':None,'gap':None,'reason':'invalid_answer','box':None,'control':None}
 if full is None:return result
 location=v.predict([image],c['settings']['localization_prompt']);grid=w.last_input_info['image_grid_thw'];assert len(grid)==1
 _,gh,gw=grid[0];rs=(gw*patch,gh*patch)
 tight=pixel_box(location,image.size,rs,c['settings']);box=band(tight,image.size)
 result.update(localization_raw=location,processor_size=rs,tight_box=tight,box=box,reason='invalid_region')
 if box is None:return result
 crop_raw=v.predict([image.crop(box)],prompt);crop=norm(crop_raw)
 result.update(crop_raw=crop_raw,crop_prediction=crop,reason='crop_disagreement')
 if crop!=full:return result
 result['reason']='missing_control';control=control_box(box,image.size);result['control']=control
 if control is None:return result
 sc=v.score(mask(image,control,c['settings']['mask_value']),prompt,full)
 se=v.score(mask(image,box,c['settings']['mask_value']),prompt,full)
 assert math.isfinite(sc) and math.isfinite(se)
 result.update(score_control=sc,score_evidence=se,gap=sc-se,reason='scored')
 return result

rows=[];total=sum(1+len(d['attacks']) for d in s['documents'])
try:
 for di,d in enumerate(s['documents']):
  cases=[{'family':'clean','epsilon':0,'images':d['images']}]+d['attacks']
  for ai,case in enumerate(cases):
   key=f'case_{di:02d}_{ai:02d}.json';target=out/key;item=case['images'][0]
   if target.exists():
    row=json.loads(target.read_text());assert row['record_id']==d['record_id'] and row['family']==case['family'] and row['epsilon']==case['epsilon'] and row['input_sha256']==item['sha256'];rows.append(row);print('RESUME',key,flush=True);continue
   path=root/'staging'/item['path'].lstrip('/');assert digest(path)==item['sha256']
   image=Image.open(path).convert('RGB');start=time.monotonic();obs=observe(image,d['dataset_name']);full=obs['full_prediction']
   buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=85);buffer.seek(0)
   jpeg=normalize_receipt_total(v.predict([Image.open(buffer).convert('RGB')],prompt),d['dataset_name'])
   agrees=full is not None and obs['crop_prediction']==full
   predictions={'none':full,'jpeg':jpeg,'band_agreement':full if agrees else None,'band_csev':full if agrees and obs['gap'] is not None and obs['gap']>=0 else None}
   methods={k:{'prediction':pred,'accepted':pred is not None,'correct':pred is not None and pred==d['clean_target'],'targeted':pred is not None and pred==d['frozen_attack_target']} for k,pred in predictions.items()}
   row={'record_id':d['record_id'],'family':case['family'],'epsilon':case['epsilon'],'input_sha256':item['sha256'],'observation':obs,'methods':methods,'elapsed_seconds':time.monotonic()-start}
   write(target,row);rows.append(row)
   print(f'{len(rows)}/{total} {d["record_id"]} {case["family"]} eps={case["epsilon"]} '+str({k:val['prediction'] for k,val in methods.items()}),flush=True)
 # Report denominators explicitly: all tested cases in each family/budget.
 groups=collections.defaultdict(list)
 for row in rows:groups[f'{row["family"]}/eps{row["epsilon"]}'].append(row)
 summary={'total_cases':len(rows),'groups':{},'limitations':protocol['limitations']}
 for group,rs in groups.items():
  summary['groups'][group]={'n':len(rs),'methods':{}}
  for name in ['none','jpeg','band_agreement','band_csev']:
   summary['groups'][group]['methods'][name]={metric:sum(r['methods'][name][metric] for r in rs) for metric in ['accepted','correct','targeted']}
 write(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
finally:w.unload()
