import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch
from PIL import Image
B=Path('/Users/prabudhdkandpal/Desktop/CSCI566/test');out=B/'method_figure'
r=json.loads((B/'receipt_bands_full_v1_development_review/development/7f1cff32dd6ae78ceffcaca2f2e1f68a4d807f43210f98c82eaae23edaaa92ac.json').read_text())
s=json.loads((B/'csev_colab_validation/subset.json').read_text());d=next(d for d in s['documents'] if d['record_id']==r['record_id']);a=next(a for a in d['attacks'] if a['family']==r['family'] and a['epsilon']==r['epsilon']);p=B/'csev_colab_validation/staging'/a['images'][0]['path'].lstrip('/')
assert hashlib.sha256(p.read_bytes()).hexdigest()==a['images'][0]['sha256']==r['input_sha256']
im=Image.open(p).convert('RGB');o=r['observation']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(18,7.5),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ink='#172B42';muted='#526477';blue='#2874D0';teal='#008C83';light='#F3F6FA';orange='#B76C22'
def text(x,y,s,size=12,color=ink,weight='normal',ha='left',va='center'):
 return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,transform=ax.transAxes)
def box(x,y,w,h,fc=light,ec='none',lw=1.3):
 q=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',facecolor=fc,edgecolor=ec,linewidth=lw,transform=ax.transAxes);ax.add_patch(q);return q
def arrow(a,b,color=muted):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,linewidth=1.6,color=color,transform=ax.transAxes))
def view(rect,image):
 z=fig.add_axes(rect);z.imshow(image);z.axis('off');return z
text(.035,.94,'Contextual-band verified recovery',25,weight='bold')
text(.035,.892,'Read two contextual views of the predicted total region; return their shared answer or abstain.',13,color=muted)
for x,num,title in [(.035,'1','Localize'),(.265,'2','Preserve context'),(.585,'3','Read separately'),(.805,'4','Compare answers')]:
 box(x,.81,.023,.037,fc=ink);text(x+.0115,.828,num,11,'white','bold','center');text(x+.034,.828,title,14,weight='bold')
# Receipt and exact saved regions.
z=view([.038,.215,.158,.55],im)
for coords,color,width,style in [(o['expanded_box'],teal,2.3,'--'),(o['box'],blue,2.3,'-'),(o['tight_box'],orange,2,':')]:
 l,t,rr,b=coords;z.add_patch(Rectangle((l,t),rr-l,b-t,fill=False,edgecolor=color,linewidth=width,linestyle=style))
text(.035,.185,'Attacked receipt',12,weight='bold')
text(.035,.151,'Model predicts a region.',10.5,muted)
text(.035,.125,'Boxes mapped to image pixels.',10.5,muted)
# Crops, full width unchanged from saved observations.
box(.263,.499,.274,.263,fc='#EEF4FD',ec='#CEE0F7')
text(.277,.735,'Evidence band',12,blue,'bold')
view([.277,.536,.246,.174],im.crop(o['box']))
box(.263,.17,.274,.29,fc='#ECF7F5',ec='#C9E5DF')
text(.277,.433,'Expanded context band',12,teal,'bold')
view([.277,.205,.246,.204],im.crop(o['expanded_box']))
arrow((.202,.54),(.25,.63),blue);arrow((.202,.49),(.25,.32),teal)
# Two calls to same frozen model.
for y,col,raw in [(.63,blue,o['crop']),(.32,teal,o['wide'])]:
 arrow((.545,y),(.581,y),col)
 box(.593,y-.075,.135,.15,fc='white',ec=col,lw=1.5)
 text(.6605,y+.035,'Document reader',12,ink,'bold','center')
 text(.6605,y-.025,raw,22,col,'bold','center')
text(.66,.185,'Same frozen model',10.5,muted,ha='center')
text(.66,.155,'Separate image inputs',10.5,muted,ha='center')
# Compare, then branches. Both inputs converge visibly.
arrow((.738,.63),(.788,.50),blue);arrow((.738,.32),(.788,.455),teal)
box(.798,.421,.164,.118,fc=light,ec='#D5DEE7')
text(.88,.497,'Normalize + compare',12,weight='bold',ha='center')
text(.88,.456,'Do the answers agree?',11,muted,ha='center')
arrow((.88,.55),(.88,.645),teal);text(.894,.594,'Yes',10.5,teal)
box(.798,.655,.164,.105,fc='#E4F4EE',ec='#AED9C7')
text(.88,.719,'Return 29.96',18,teal,'bold','center')
text(.88,.679,'Shared band answer',10.5,muted,ha='center')
arrow((.88,.410),(.88,.305),muted);text(.894,.358,'No / missing',10,muted)
box(.798,.195,.164,.10,fc='#F5F2ED',ec='#DDD4C7')
text(.88,.257,'Abstain',17,ink,'bold','center')
text(.88,.219,'Refer for review',10.5,muted,ha='center')
# Example-specific reference is not connected to decision path.
ax.plot([.035,.965],[.103,.103],color='#DCE3EB',lw=1)
text(.035,.075,'QWEN EXAMPLE',9.5,muted,'bold')
text(.19,.075,'No defense: 299.60    |    Correct total: 29.96    |    Standard attack, epsilon = 16/255',11,ink)
text(.035,.035,'The full-image answer and annotated total are shown for illustration only; neither is required by the recovery decision. Agreement does not guarantee correctness.',10,muted)
for suffix in ['png','pdf','svg']:fig.savefig(out/f'contextual_band_recovery_draft.{suffix}',dpi=200,facecolor='white')
(out/'example_provenance.json').write_text(json.dumps({'record_id':r['record_id'],'family':r['family'],'epsilon_255':r['epsilon'],'input_sha256':r['input_sha256'],'saved_observation':o,'figure_note':'Selected development example, not aggregate evidence. All crop geometry and displayed predictions are from the saved record.'},indent=2))
(out/'caption.txt').write_text('Contextual-band verified recovery. A vision-language model localizes the receipt total region. We expand it into a full-width evidence band and a wider context band, read the two views separately, and return their normalized shared answer; disagreement or missing evidence triggers abstention. The illustrated saved development example recovers 29.96 from an attacked receipt whose full-image prediction is 299.60. The annotated total is used only to assess the result, not by the defense. Agreement is not a guarantee of correctness.')
