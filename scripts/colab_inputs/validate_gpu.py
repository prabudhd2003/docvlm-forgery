"""Diagnostic only: no CARC submissions, no production outputs, no calibrated claims."""
import os,sys,json,hashlib,platform,importlib.metadata,subprocess,traceback,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/src'))
os.environ['HF_HOME']=str(ROOT/'hf_cache')
RESULTS=ROOT/'validation_results'; RESULTS.mkdir(exist_ok=True)
config=json.loads((ROOT/'csev_receipts_v1.json').read_text())
subset=json.loads((ROOT/'subset.json').read_text())
def local(remote):return ROOT/'staging'/remote.lstrip('/')
def save(name,value):(RESULTS/name).write_text(json.dumps(value,indent=2,default=str))
def main():
 import torch,numpy as np
 from PIL import Image
 from models.qwen import QwenVL
 from csev.defense import Verifier,control_box,mask
 from csev.runner import comparisons
 from csev.adaptive import gradient_step
 assert torch.cuda.is_available(),'Select a Colab GPU runtime.'
 versions={k:platform.python_version() if k=='python' else importlib.metadata.version(k) for k in config['runtime_versions']}
 mismatch={k:[config['runtime_versions'][k],v] for k,v in versions.items() if config['runtime_versions'][k]!=v}
 save('environment.json',dict(versions=versions,mismatches=mismatch,gpu=torch.cuda.get_device_name(),vram_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=torch.version.cuda))
 assert not mismatch,f'Environment mismatch: {mismatch}'
 for remote,h in subset['files'].items():
  assert hashlib.sha256(local(remote).read_bytes()).hexdigest()==h,remote
 ocr=config['legacy']['ocr']; ocr['executable']='/usr/bin/tesseract'; ocr['tessdata_dir']=str(local(ocr['tessdata_dir']))
 actual=subprocess.check_output([ocr['executable'],'--version'],text=True).splitlines()[0]
 save('ocr_environment.json',dict(carc_version=ocr['version'],colab_version=actual,exact_binary_match=hashlib.sha256(Path(ocr['executable']).read_bytes()).hexdigest()==ocr['executable_sha256'],note='Same traineddata; Colab OCR binary and operating system may differ.'))
 wrapper=QwenVL(device=torch.device('cuda'),**config['legacy']['model'])
 prompt=config['legacy']['prompts']['receipt']; settings=config['settings']
 verifier=Verifier(wrapper,settings,prompt)
 torch.cuda.reset_peak_memory_stats()
 total=sum(1+len(d['attacks']) for d in subset['documents']); completed=0
 # All nine comparisons run on all clean and attacked images. tau=0 is diagnostic only.
 for di,d in enumerate(subset['documents']):
  for ai,a in enumerate([dict(family='clean',epsilon=0,images=d['images'])]+d['attacks']):
   image=Image.open(local(a['images'][0]['path'])).convert('RGB')
   start=time.monotonic()
   print(f"Starting {completed+1}/{total}: {d['record_id']} {a['family']} epsilon={a['epsilon']}",flush=True)
   rows,obs=comparisons(image,d,verifier,wrapper,config,0.0)
   assert len(rows)==len(settings['conditions'])
   save(f'case_{di:02d}_{ai:02d}.json',dict(record_id=d['record_id'],family=a['family'],epsilon=a['epsilon'],conditions=rows,observation=obs,diagnostic_threshold=0,elapsed=time.monotonic()-start))
   completed+=1
   print(f'PASS {completed}/{total}; CSEV outcome={obs["reason"]}',flush=True)
 # Exercise full-image, crop, and both mask gradient branches on three receipts.
 # Fixed geometry and high hinge threshold are branch tests, not proposed defense decisions.
 for di,d in enumerate(subset['documents'][:3]):
  image=Image.open(local(d['images'][0]['path'])).convert('RGB'); w,h=image.size
  box=(0,0,max(28,w//3),max(28,h//3)); control=control_box(box,image.size)
  assert control is not None
  scores=[verifier.score(mask(image,b,settings['mask_value']),prompt,d['frozen_attack_target']) for b in [box,control]]
  assert all(math.isfinite(v) for v in scores)
  clean=np.asarray(image)
  updated,loss=gradient_step(wrapper,image,clean,d['frozen_attack_target'],prompt,dict(box=box,control=control),1e6,8,settings)
  delta=int(np.abs(updated.astype(int)-clean.astype(int)).max()); assert delta<=8 and math.isfinite(loss)
  attacked=Image.fromarray(updated); attacked.save(RESULTS/f'gradient_{di}.png')
  replay=verifier(attacked,d['dataset_name'])
  save(f'gradient_{di}.json',dict(loss=loss,linf_255=delta,masked_scores=scores,replay=replay,note='One diagnostic update, synthetic boxes and forced hinge; not a full adaptive attack.'))
  print(f'PASS adaptive gradient and replay {di+1}/3',flush=True)
 save('summary.json',dict(status='PASS',completed_cases=completed,condition_records=completed*len(settings['conditions']),gradient_tests=3,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,limitations=['No production scheduling, shard merge, full calibration, adaptive search or checkpoint/resume tested.','Diagnostic threshold 0; not a defense effectiveness study.','Subset not representative and does not cover the largest possible image.','Colab drivers, OS, GPU and OCR binary differ from CARC.']))
 print('PASS: GPU function validation finished. This does not guarantee production completion.',flush=True)
 wrapper.unload()
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save('failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
  raise
