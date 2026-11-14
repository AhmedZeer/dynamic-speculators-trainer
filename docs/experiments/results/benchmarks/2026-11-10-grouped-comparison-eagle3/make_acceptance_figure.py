# Reference script, not directly runnable here: it ran in the analysis session's
# scratchpad with helper plot.py (shared color constants) alongside it, and
# dynamism_ttt3.csv from the sibling ttt3 benchmark directory. Paths below are
# relative to that original scratchpad layout, kept for provenance.
import numpy as np, json, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE,ORANGE,INK,MUTED,GRID='#2a78d6','#eb6834','#1f1f1e','#6b6a63','#e4e3de'
plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
 'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
j=json.load(open('../2026-11-10-matched-comparison-v2-acceptance/summary.json'))['results']
def acc(model,cat):
    a=j[model]['nemotron/'+cat]['acceptance']
    return 100*a['num_accepted_tokens']/a['num_draft_tokens']
d=pd.read_csv('../2026-11-10-attention-scaling-dynamism-v1-ttt3/dynamism_ttt3.csv')
d6=d[d.steps==6000].set_index('category')['correct_acceptance_percent']
cats=[('nemo-math','Math'),('nemo-stem','STEM'),('nemo-chat','Chat')]
arms=[('Vanilla Eagle3 (2026-11-10)',[47.02,40.09,36.05],'#8a8983'),
      ('Static LoRA (V/O)',[acc('trained-lora-c1',c) for c,_ in cats],ORANGE),
      ('V/O generator, 24K / 4K',[acc('generator-4000step-c1',c) for c,_ in cats],'#1baf7a'),
      ('Attention generator, 24K / 4K',[67.93,52.49,40.45],BLUE),
      ('Attention generator, 100K / 6K',[d6[c.replace('nemo-','')] for c,_ in cats],'#184f95')]
rows=[]; fig,ax=plt.subplots(figsize=(11,5)); w=0.165; x=np.arange(3)
for i,(lab,v,clr) in enumerate(arms):
    xs=x+(i-2)*w
    ax.bar(xs,v,w-0.03,color=clr,label=lab,zorder=3)
    for xi,s in zip(xs,v): ax.text(xi,s+0.4,f'{s:.2f}%',ha='center',va='bottom',fontsize=8,color=INK)
    rows+=[dict(arm=lab,category=n,draft_acceptance_percent=s) for (c,n),s in zip(cats,v)]
ax.set_xticks(x,[n for _,n in cats]); ax.set_ylim(30,80); ax.set_ylabel('Draft acceptance rate (%)')
ax.grid(axis='x',visible=False)
ax.legend(loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.2),fontsize=9)
fig.text(0.5,0.01,"Accepted ÷ proposed draft tokens, 3 draft tokens/cycle, C1, 50 prompts/category. Vanilla Eagle3: original (unmodified) Eagle3\n"
 "drafter, no adapter, measured today on the frozen 100K/TTT3 workload. Static LoRA / V/O generator: matched-suite comparison-v2 run supplied\n"
 "2026-11-10 (fills the acceptance gap in GEN-003/extended_acceptance_table.md). Attention 24K/4K: GEN-011. Attention 100K/6K: today's TTT3 suite,\n"
 "correct-prompt policy. Arms use four different runs/workloads; single runs, no CIs; code category not measured for the two attention arms.",
 ha='center',fontsize=7.6,color=MUTED)
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig('grouped_acceptance_vs_eagle3.png',dpi=200); fig.savefig('grouped_acceptance_vs_eagle3.pdf')
pd.DataFrame(rows).to_csv('grouped_acceptance_vs_eagle3.csv',index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
