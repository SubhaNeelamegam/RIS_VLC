import json,sys; sys.path.insert(0,'.')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from geom import gains,iy,iz
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.grid':True,'grid.alpha':.3,'figure.dpi':100,'savefig.dpi':300,'legend.fontsize':7.5})
R=json.load(open('res.json')); snr=np.array(R['snr']); fine=np.array(R['fine'])
C='#c0392b';B='#1f3b57';G='#2e8b57';O='#d68910';P='#7d3c98';K='#555555'
def cl(x): x=np.array(x,float); x[x<1e-12]=np.nan; return x
# Fig1 scenario
fig,ax=plt.subplots(figsize=(5.4,3.0))
ax.set_xlim(0,5);ax.set_ylim(0,3.3);ax.set_aspect('equal');ax.add_patch(Rectangle((0,0),5,3,fill=False,lw=1.2))
ax.plot(2.5,3.0,'o',ms=12,color='#f0b429');ax.text(2.5,3.12,'LED (2.5, 3.0 m)',ha='center',fontsize=8)
ax.add_patch(Rectangle((4.96,1.0),0.08,1.0,color=B));ax.text(4.9,2.1,'RIS panel\n16×16, 4 groups',ha='right',fontsize=8)
ax.plot(1.0,0.85,'s',color='k',ms=7);ax.text(1.0,0.55,'PD (1.0, 0.85 m)',ha='center',fontsize=8)
ax.plot([2.5,1.0],[3.0,0.85],'--',color=C,lw=1);ax.text(1.35,1.9,'LOS (blocked)',color=C,fontsize=8,rotation=55)
ax.plot([2.5,4.96],[3.0,1.5],color=K,lw=.8);ax.plot([4.96,1.0],[1.5,0.85],color=K,lw=.8)
ax.set_xlabel('x (m), side view (y = 2.5 m)');ax.set_ylabel('z (m)');ax.grid(False)
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f1.png")
plt.close()
#fig.tight_layout();fig.savefig('D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f1.png');plt.close()
# Fig2 partitions
h,ok=gains(); idx=np.where(ok)[0]
A={'Balanced equal':np.load('A_bal.npy'),'Contiguous (rows)':np.load('A_con.npy'),'Stage 1':np.load('A_s1.npy'),'Stage 1+2 (proposed)':np.load('A_s2.npy')}
cols=['#1f3b57','#2e8b57','#d68910','#c0392b']
fig,axs=plt.subplots(1,4,figsize=(7.2,2.2))
for ax,(n,a) in zip(axs,A.items()):
    M=np.full((16,16),-1)
    for k,e in enumerate(idx): M[iz[e],iy[e]]=a[k]
    cm=matplotlib.colors.ListedColormap(['#dddddd']+cols); ax.imshow(M+1,origin='lower',cmap=cm,vmin=0,vmax=4)
    ax.set_title(n,fontsize=7.5);ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f2.png")
plt.close()
#fig.tight_layout();fig.savefig('figs/f2.png');plt.close()
# Fig3 BER SM
cc=R['curves']['c']
fig,ax=plt.subplots(figsize=(5.2,3.6))
sty=[('Balanced equal',B,'o','-'),('Random equal',O,'s','-'),('Contiguous',G,'D','-'),('Unequal size',P,'v','-'),('Stage 1',K,'^','-'),('Stage 1+2',C,'^','-')]
for n,c,m,l in sty: ax.semilogy(snr,cl(cc[n]),color=c,marker=m,ls=l,ms=4,label={'Stage 1':'Proposed Stage 1','Stage 1+2':'Proposed Stage 1+2','Contiguous':'Contiguous rows','Unequal size':'Unequal size (random fill)'}.get(n,n))
ax.semilogy(R['fine'],cl(R['curves']['f']['16-PAM']),color='k',ls='--',lw=1,label='16-PAM, all elements (reference)')
ax.set_ylim(1e-11,1);ax.set_xlim(0,40);ax.set_xlabel('SNR (dB)');ax.set_ylabel('BER');ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f3.png")
plt.close()
#fig.tight_layout();fig.savefig('figs/f3.png');plt.close()
# Fig4 GSM
fig,ax=plt.subplots(figsize=(5.2,3.4))
for n,c,m in [('Balanced equal',B,'o'),('Contiguous',G,'D'),('Stage 1',K,'^'),('Stage 1+2',C,'^')]:
    ax.semilogy(snr,cl(R['gsm'][n]),color=c,marker=m,ms=4,label={'Stage 1':'Proposed Stage 1','Stage 1+2':'Proposed Stage 1+2 (SM-tuned)','Contiguous':'Contiguous rows'}.get(n,n))
ax.set_ylim(1e-11,1);ax.set_xlim(0,40);ax.set_xlabel('SNR (dB)');ax.set_ylabel('BER');ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f4.png")
plt.close()
#fig.tight_layout();fig.savefig('figs/f4.png');plt.close()
# Fig5 refine + UB
fig,ax=plt.subplots(figsize=(5.2,3.4))
ax.semilogy(snr,cl(cc['Balanced equal']),color=B,marker='o',ms=4,label='Balanced equal')
ax.semilogy(snr,cl(cc['Stage 1']),color=K,marker='^',ms=4,label='Stage 1 (simulated)')
ax.semilogy(snr,cl(cc['Stage 1+2']),color=C,marker='^',ms=4,label='Stage 1+2 (simulated)')
ax.semilogy(snr,cl(R['ub']['Stage 1']),color=K,ls=':',label='Union bound, Stage 1')
ax.semilogy(snr,cl(R['ub']['Stage 1+2']),color=C,ls=':',label='Union bound, Stage 1+2')
ax.set_ylim(1e-11,3);ax.set_xlim(0,40);ax.set_xlabel('SNR (dB)');ax.set_ylabel('BER');ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f5.png")
plt.close()
#fig.tight_layout();fig.savefig('figs/f5.png');plt.close()
# Fig6 sensitivity
s=R['sens']; r=s['r']
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0))
axs[0].semilogy(r,s['s1_30'],color=K,marker='^',ls='--',ms=4,label='Stage 1, 30 dB');axs[0].semilogy(r,s['s2_30'],color=C,marker='o',ms=4,label='Stage 1+2, 30 dB')
axs[0].semilogy(r,s['s2_25'],color=O,marker='s',ms=4,label='Stage 1+2, 25 dB');axs[0].set_xlabel('Stagger ratio r');axs[0].set_ylabel('BER');axs[0].legend()
axs[1].plot(r,s['dmin1'],color=K,marker='^',ls='--',ms=4,label='Stage 1');axs[1].plot(r,s['dmin2'],color=C,marker='o',ms=4,label='Stage 1+2');axs[1].set_xlabel('Stagger ratio r');axs[1].set_ylabel('Weighted $d_{min}$ (×10$^{-12}$)');axs[1].legend()
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f6.png")
plt.close()
#fig.tight_layout();fig.savefig('figs/f6.png');plt.close()
# Fig7 failures
f=R['fail']['30'];p=R['fail']['p']
fig,ax=plt.subplots(figsize=(5.2,3.3))
for k,c,m,l in [('frozen_bal',B,'o','Frozen balanced equal'),('frozen_con',G,'D','Frozen contiguous rows'),('frozen_prop',O,'s','Frozen proposed'),('repart',C,'^','Re-partitioned proposed')]:
    mu=np.array([x[0] for x in f[k]]);e=np.array([x[1] for x in f[k]]);ax.errorbar(p,mu,yerr=e,color=c,marker=m,ms=4,capsize=2,label=l)
ax.set_yscale('log');ax.set_xlabel('Failed RIS elements (%)');ax.set_ylabel('BER at 30 dB');ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f7.png")
plt.close()
#fig.tight_layout();fig.savefig('figs/f7.png');plt.close()
# Fig8 CSI
cs=R['csi'];fig,ax=plt.subplots(figsize=(5.0,3.2))
for k,c,m,l in [('s1',K,'^','Stage 1'),('prop',C,'o','Stage 1+2')]:
    mu=np.array([x[0] for x in cs[k]]);e=np.array([x[1] for x in cs[k]]);ax.errorbar(cs['e'],mu,yerr=e,color=c,marker=m,ms=4,capsize=2,label=l)
ax.axhline(R['contig_rows']['ber30'],color=G,ls='--',label='Contiguous rows (geometry only)')
ax.set_yscale('log');ax.set_xlabel('Relative error of element-gain estimates (%, std)');ax.set_ylabel('BER at 30 dB');ax.legend()
fig.tight_layout()
fig.savefig(r"D:\College\2026 -27 Odd\my\spatialmodulation\New folder\f8.png")
plt.close()
#fig.tight_layout();fig.savefig('figs/f8.png');plt.close()
