# Reference script, not directly runnable here: see make_acceptance_figure.py's
# header. speedup_ttt3.csv is in the sibling ttt3 benchmark directory.
import numpy as np, json, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE,ORANGE,INK,MUTED,GRID='#2a78d6','#eb6834','#1f1f1e','#6b6a63','#e4e3de'
plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
 'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
B='/home/user/dynamic-speculators-trainer/docs/experiments/results/benchmarks/'
M=json.load(open(B+'2026-10-22-comparison-v2/metrics.json'))['models']
for x in M: print(x['name'],{k:v for k,v in x.items() if k not in('output_tokens_per_second','name')})
m={x['name']:x['output_tokens_per_second'] for x in M}
att={r['model']:r for r in json.load(open(B+'2026-10-31-attention-24k-6k-c1/summary.json'))['rows']}
print(list(att))
cats=[('nemo-math','Math'),('nemo-code','Code'),('nemo-stem','STEM'),('nemo-chat','Chat')]
a4=[r for k,r in att.items() if '4000' in k][0]
arms=[('Vanilla Eagle3',[m['eagle3-base'][c] for c,_ in cats],'#8a8983'),
      ('Static LoRA (V/O)',[m['trained-lora'][c] for c,_ in cats],ORANGE),
      ('V/O generator, 24K / 4K',[m['generator-4000step'][c] for c,_ in cats],'#1baf7a'),
      ('Attention generator, 24K / 4K',[a4['nemotron/'+c] for c,_ in cats],BLUE)]
base=np.array(arms[0][1])
sp=pd.read_csv('../2026-11-10-attention-scaling-dynamism-v1-ttt3/speedup_ttt3.csv')
g=lambda mdl,pol,st,c: sp[(sp.model==mdl)&(sp.policy==pol)&(sp.steps==st)&(sp.category==c)].tokens_per_second.iloc[0] if c!='code' else np.nan
c100=[g('generator','correct',6000,c.split('-')[1]) for c,_ in cats]
d100=[g('original_drafter','none',0,c.split('-')[1]) for c,_ in cats]
arms.append(('Attention generator, 100K / 6K',c100,'#184f95'))
rows=[]
fig,ax=plt.subplots(figsize=(11,5)); w=0.165; x=np.arange(4)
for i,(lab,v,clr) in enumerate(arms):
    den=np.array(d100) if '100K' in lab else base
    sp=np.array(v)/den; xs=x+(i-2)*w
    ax.bar(xs,sp,w-0.03,color=clr,label=lab,zorder=3)
    for xi,s in zip(xs,sp):
        if np.isnan(s): ax.text(xi,0.905,'not\nmeasured',ha='center',va='bottom',fontsize=7,color=MUTED); continue
        ax.text(xi,s+0.008,f'{s:.2f}×',ha='center',va='bottom',fontsize=8,color=INK)
    rows+= [dict(arm=lab,category=n,tokens_per_second=t,x_vs_vanilla_eagle3=s) for (c,n),t,s in zip(cats,v,sp)]
ax.set_xticks(x,[n for _,n in cats]); ax.set_ylim(0.9,1.32); ax.set_ylabel('Speedup vs vanilla Eagle3 (×)')
ax.axhline(1,color=MUTED,lw=1,zorder=2); ax.grid(axis='x',visible=False)
ax.legend(loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.2),fontsize=9)
fig.text(0.5,0.01,'Output tok/s ÷ vanilla Eagle3 from the same run. Eagle3, static LoRA, V/O 24K: GEN-003. Attention 24K: GEN-011 (÷ GEN-003 Eagle3).\nAttention 100K/6K: 2026-11-10 run ÷ same-day Eagle3 on the same prompts; that run has no code workload. Single runs, no CIs.',ha='center',fontsize=8,color=MUTED)
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig('grouped_speedup_vs_eagle3.png',dpi=200); fig.savefig('grouped_speedup_vs_eagle3.pdf')
pd.DataFrame(rows).to_csv('grouped_speedup_vs_eagle3.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
