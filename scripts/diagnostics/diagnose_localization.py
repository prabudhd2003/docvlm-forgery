"""Development-only coordinate/localization diagnostic; no production edits."""
import argparse,json,sys,copy,math,hashlib
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/'source/src'))
import torch
from PIL import Image,ImageDraw
from models.qwen import QwenVL
from csev.defense import Verifier,parse_box,control_box,mask
from normalization import normalize_receipt_total
c=json.loads((root/'csev_receipts_v1.json').read_text());subset=json.loads((root/'subset.json').read_text())
(out/'protocol.json').write_text(json.dumps({'purpose':'Development diagnostic, not final evaluation','git_commit':c['git_commit'],'padding_fraction':.03,'documents':[d['record_id'] for d in subset['documents']]},indent=2))
w=QwenVL(device=torch.device('cuda'),**c['legacy']['model']);prompt=c['legacy']['prompts']['receipt'];v=Verifier(w,c['settings'],prompt)
patch=w.processor.image_processor.patch_size
assert isinstance(patch,int),f'Unexpected patch size: {patch}'
def loc(im,text):
 settings=copy.deepcopy(c['settings']);settings['localization_prompt']=text
 vv=Verifier(w,settings,prompt);raw=vv.predict([im],text)
 grid=copy.deepcopy(w.last_input_info['image_grid_thw'])
 assert len(grid)==1
 _,gh,gw=grid[0]
 return raw,(gw*patch,gh*patch),grid

def pixel_box(raw,size,resized):
 # Convert declared resized-image pixels to the existing normalized parser.
 try:
  value=json.loads(raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())['box']
  if not isinstance(value,list) or len(value)!=4:return None
  if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):return None
  normalized=[x*1000/d for x,d in zip(value,(*resized,*resized))]
  return parse_box(json.dumps({'box':normalized}),size,c['settings'])
 except (ValueError,TypeError,KeyError):return None

def padded(box,size):
 if box is None:return None
 W,H=size;dx=round(W*.03);dy=round(H*.03);l,t,r,b=box
 return (max(0,l-dx),max(0,t-dy),min(W,r+dx),min(H,b+dy))
records=[]
try:
 for i,d in enumerate(subset['documents']):
  item=d['images'][0];path=root/'staging'/item['path'].lstrip('/')
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  im=Image.open(path).convert('RGB');ds=d['dataset_name']
  full_raw=v.predict([im],prompt);full=normalize_receipt_total(full_raw,ds)
  old_raw,rs,grid=loc(im,c['settings']['localization_prompt'])
  native_prompt=('Locate the final transaction total amount AND its label on this receipt. '
    'Read the receipt to distinguish the final payable total from subtotal, tax, item prices, cash tendered and change. '
    f'The processed image is {rs[0]} pixels wide and {rs[1]} pixels high. '
    'Return only JSON {"box":[left,top,right,bottom]} with absolute pixel coordinates on that processed image. '
    'Include both the final total label and its amount. Do not normalize coordinates. '
    'If the region cannot be found, return {"box":null}.')
  new_raw,new_rs,new_grid=loc(im,native_prompt)
  newbox=pixel_box(new_raw,im.size,new_rs)
  boxes={'original_normalized':parse_box(old_raw,im.size,c['settings']),
         'original_as_pixels':pixel_box(old_raw,im.size,rs),
         'native_pixels':newbox,'native_pixels_padded':padded(newbox,im.size)}
  row={'record_id':d['record_id'],'clean_target':d['clean_target'],'full_raw':full_raw,'full_prediction':full,
       'original_localization':old_raw,'native_localization':new_raw,'native_prompt':native_prompt,
       'image_size':im.size,'processor_size':rs,'native_processor_size':new_rs,'grid':grid,'native_grid':new_grid,'variants':{}}
  for name,box in boxes.items():
   result={'box':box,'raw':None,'prediction':None,'agrees':False}
   if box:
    crop=im.crop(box);raw=v.predict([crop],prompt);pred=normalize_receipt_total(raw,ds)
    result.update(raw=raw,prediction=pred,agrees=pred is not None and pred==full,correct=pred==d['clean_target'])
    crop.save(out/f'{i:02d}_{name}_crop.png')
    overlay=im.copy();ImageDraw.Draw(overlay).rectangle(box,outline='red',width=max(3,im.width//200));overlay.thumbnail((800,1200));overlay.save(out/f'{i:02d}_{name}_overlay.png')
    if name=='native_pixels_padded' and result['agrees']:
     control=control_box(box,im.size);result['control']=control
     if control:
      sc=v.score(mask(im,control,c['settings']['mask_value']),prompt,full)
      se=v.score(mask(im,box,c['settings']['mask_value']),prompt,full)
      result.update(score_control=sc,score_evidence=se,gap=sc-se)
   row['variants'][name]=result
  (out/f'clean_{i:02d}.json').write_text(json.dumps(row,indent=2))
  records.append(row)
  print(f'{i+1}/15 {d["record_id"]} full={full} target={d["clean_target"]} '+str({k:r['prediction'] for k,r in row['variants'].items()}),flush=True)
 summary={'documents':len(records),'full_correct':sum(r['full_prediction']==r['clean_target'] for r in records),
          'variants':{name:{'valid_boxes':sum(r['variants'][name]['box'] is not None for r in records),
                            'agreement':sum(r['variants'][name]['agrees'] for r in records),
                            'correct_crops':sum(r['variants'][name].get('correct',False) for r in records)} for name in boxes},
          'note':'No best variant selected automatically; inspect boxes and use separate held-out evaluation.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
finally:w.unload()
