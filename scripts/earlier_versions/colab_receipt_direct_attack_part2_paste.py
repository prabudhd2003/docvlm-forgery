# PART 2 of the frozen recovery-only direct attack: receipts 6-10 of the frozen list.
# Paste as ONE new cell at the end of CSEV_Full_Colab_Study.ipynb, after cells 1-11 have run
# (mount, extract, runner, environment, run_stage helper, preflight). Do not enable any RUN_* flag.
import os, subprocess, threading, json
from datetime import datetime
os.environ['MPLBACKEND']='Agg'
assert 'csev_full_colab' in str(OUT), f'Wrong Drive folder: {OUT}. Fix DRIVE_FOLDER in the mount cell.'
assert ROOT.is_dir() and (OUT/'frozen.json').exists() and (OUT/'evaluation').is_dir()
before={p.parent.name for p in (OUT/'adaptive').glob('*/result.json')}
assert len(before)>=25, f'Expected the 25 finished runs in {OUT}/adaptive, found {len(before)}'
if 'process' in globals():
    assert process.poll() is not None, 'A previous child process is still running.'
LAUNCHER_SOURCE=r'''"""Scheduling-only continuation, part 2, of the frozen Qwen study; no method changes."""
import argparse, hashlib, importlib.util, inspect, json, textwrap
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.out)
spec=importlib.util.spec_from_file_location('frozen_study',root/'full_study.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=m.Study(root,out)  # Verifies the original runner, source, input and policy contract.
f=s.frozen()
first=m.read(out/'recovery_budget_schedule.json')
assert first['original_contract']==s.sig and first['document_ids']==f['adaptive_ids'][:5]
schedule={'original_contract':s.sig,'method':'verified_recovery','document_ids':f['adaptive_ids'][5:10],'epsilons':[4,8],'restarts':[0,1],'max_runs':20,'wall_seconds_per_session':14400,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':'IDs six to ten in the original frozen order, completing the frozen ten-receipt list; not selected on outcomes. Only scheduling changes; original gradient, query, replay and checkpoint code retained.'}
manifest=out/'recovery_budget_schedule_part2.json'
if manifest.exists():assert m.read(manifest)==schedule,'Existing part-2 schedule differs.'
else:m.write(manifest,schedule)
source=textwrap.dedent(inspect.getsource(m.Study.adaptive))
needle="for rid in ids:\n"
assert source.count(needle)==1
source=source.replace(needle,"for rid in ids[5:10]:\n")
needle="for method in POLICY['adaptive_methods']:\n"
assert source.count(needle)==1
source=source.replace(needle,needle+"   if method != 'verified_recovery':continue\n")
namespace={};exec(compile(source,'<recovery_schedule_part2>','exec'),m.__dict__,namespace)
print('Recovery-only schedule, part 2:',json.dumps(schedule,indent=2),flush=True)
try:
 namespace['adaptive'](s,pilot=False)
finally:
 if s.wrapper:s.wrapper.unload()
'''
launcher=ROOT/'continue_recovery_part2.py'
launcher.write_text(LAUNCHER_SOURCE)
log_path=OUT/('recovery_budget_part2_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
env=os.environ.copy();env['HF_HOME']=str(ROOT/'hf_cache')
expired=threading.Event()
with log_path.open('w') as log:
    process=subprocess.Popen([str(PYTHON),'-u',str(launcher),'--root',str(ROOT),'--out',str(OUT)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    def stop_at_budget():
        if process.poll() is None:
            expired.set();process.terminate()
    timer=threading.Timer(4*3600,stop_at_budget);timer.start()
    try:
        for line in process.stdout:
            print(line,end='');log.write(line);log.flush()
        code=process.wait()
    except BaseException:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
    finally:
        timer.cancel()
if expired.is_set():
    print('Four-hour budget reached. Finished runs and the latest checkpoint are kept; rerun this cell to continue.')
elif code:
    raise RuntimeError(f'Part-2 continuation failed: {code}. See {log_path}')
else:
    print('Part-2 schedule completed.')

# Recompute the recovery-only table from individual records (the 120-run adaptive_summary.json stays incomplete by design).
frozen_ids=json.loads((OUT/'frozen.json').read_text())['adaptive_ids']
clean={}
for p in (OUT/'evaluation').glob('*.json'):
    r=json.loads(p.read_text())
    if r['family']=='clean' and r['record_id'] in frozen_ids:
        v=r['methods']['verified_recovery'];clean[r['record_id']]=bool(v['accepted'] and v['correct'])
runs=[json.loads(p.read_text()) for p in (OUT/'adaptive').glob('*/result.json')]
runs=[r for r in runs if r['method_attacked']=='verified_recovery']
print('\nRecovery-only runs on disk:',len(runs),'of 40 planned (10 receipts x 2 budgets x 2 restarts)')
for eps in (4,8):
    done={};hit={}
    for r in runs:
        if r['epsilon']!=eps:continue
        v=r['methods']['verified_recovery'];rid=r['record_id']
        done.setdefault(rid,set()).add(r['restart'])
        hit[rid]=hit.get(rid,False) or bool(v['accepted'] and v['targeted'])
    complete=[i for i in frozen_ids if done.get(i)=={0,1}]
    eligible=[i for i in complete if clean.get(i)]
    print(f'eps {eps}: receipts with both restarts {len(complete)}/10 | success, all: {sum(hit[i] for i in complete)}/{len(complete)}'
          f' | success, clean-correct-and-accepted: {sum(hit[i] for i in eligible)}/{len(eligible)}')
    for i in frozen_ids:
        print('   ',i.split('/')[-1],'eligible' if clean.get(i) else 'clean-rejected','restarts done:',sorted(done.get(i,[])),'SUCCESS' if hit.get(i) else '')
