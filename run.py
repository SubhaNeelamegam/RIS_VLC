import sys,json,time; sys.path.insert(0,'.')
from eng import *
from scipy.stats import norm
rng=np.random.default_rng(2026)
R={}
snr=np.arange(0,41,5); fine=np.arange(0,50.01,0.25)
# ---- partitions
w15=geo_w(1.5)
A_bal=balanced_equal()
# contiguous variants
def contig(vals):
    q=np.quantile(vals,[.25,.5,.75]); return np.digitize(vals,q,right=True)
A_rows=contig(iz_a+0.001*iy_a); A_cols=contig(iy_a+0.001*iz_a)
quad=(iz_a>np.median(iz_a)).astype(int)*2+(iy_a>np.median(iy_a)).astype(int)
A_quad=quad
A_s1=stage1(w15); t=time.time(); A_s2=stage2(A_s1); R['t_stage2']=time.time()-t
t=time.time(); stage1(w15); R['t_stage1']=time.time()-t
cont={'rows':A_rows,'cols':A_cols,'quad':A_quad}
for k,a in cont.items(): R['contig_'+k]={'gains':(gsum(a)*1e10).tolist(),'ber30':ber_sm(a,30),'ber40':ber_sm(a,40),'sizes':np.bincount(a).tolist()}
bestc=min(cont,key=lambda k:ber_sm(cont[k],30)); R['best_contig']=bestc; A_con=cont[bestc]
print("contig best",bestc,{k:(round(ber_sm(a,30),4)) for k,a in cont.items()})
# ---- Table: gains and separations
def row(a):
    gs=gsum(a); return {'gains':(gs*1e10).tolist(),'gap':float(np.min(np.diff(np.sort(gs)))/gs.mean()),'dminw':float(dmin_fast(gs)),'sizes':np.bincount(a,minlength=4).tolist()}
NR=200
def avg_rand(fn):
    gaps=[];dm=[];gl=[]
    for _ in range(NR):
        a=fn(); r=row(a); gaps.append(r['gap']); dm.append(r['dminw']); gl.append(sorted(r['gains']))
    return {'gains':np.mean(gl,0).tolist(),'gap':float(np.mean(gaps)),'dminw':float(np.mean(dm))}
R['tab']={'Balanced equal':row(A_bal),'Random equal':avg_rand(lambda:random_equal(rng)),'Contiguous (%s)'%bestc:row(A_con),
 'Unequal size (random fill)':avg_rand(lambda:unequal_size(rng,w15)),'Stage 1':row(A_s1),'Stage 1+2':row(A_s2)}
for k,v in R['tab'].items(): print(k,np.round(v['gains'],2),round(v['gap'],4),v['dminw'])
# ---- BER vs SNR
def curve(a,grid): return [ber_sm(a,s) for s in grid]
def avgcurve(fn,grid,n=NR):
    c=np.zeros(len(grid))
    for _ in range(n): c+=np.array(curve(fn(),grid))
    return (c/n).tolist()
curves={}
for grid,tag in ((snr,'c'),(fine,'f')):
    curves[tag]={'Balanced equal':curve(A_bal,grid),'Random equal':avgcurve(lambda:random_equal(rng),grid,100),
     'Contiguous':curve(A_con,grid),'Unequal size':avgcurve(lambda:unequal_size(rng,w15),grid,100),
     'Stage 1':curve(A_s1,grid),'Stage 1+2':curve(A_s2,grid),
     'Unequal size+refine':None,'16-PAM':[ber_16pam(s) for s in grid],'16-PAM peak':[ber_16pam(s,True) for s in grid]}
# unequal size + refine (one draw avg 20)
us=[]
for _ in range(20): us.append(curve(stage2(unequal_size(rng,w15)),fine))
curves['f']['Unequal size+refine']=np.mean(us,0).tolist()
curves['c']['Unequal size+refine']=[float(np.interp(s,fine,curves['f']['Unequal size+refine'])) for s in snr]
R['snr']=snr.tolist(); R['fine']=fine.tolist(); R['curves']=curves
# required SNR at target BER
def need(c,t):
    c=np.array(c); i=np.where(c<=t)[0]
    if len(i)==0: return None
    i=i[0]
    if i==0: return float(fine[0])
    x0,x1=fine[i-1],fine[i]; y0,y1=np.log10(c[i-1]),np.log10(c[i]); return float(x0+(np.log10(t)-y0)*(x1-x0)/(y1-y0))
R['need']={k:{str(t):need(v,t) for t in (1e-1,1e-2,3.8e-3,1e-3)} for k,v in curves['f'].items()}
for k,v in R['need'].items(): print(k,v)
# ---- union bound
def ub(a,snr_db,hh=None):
    gs=gsum(a,hh); pts,lab=pts_sm(gs); s=sigma_for(snr_db); t=0
    for i in range(16):
        for j in range(16):
            if i!=j: t+=HD[lab[i],lab[j]]/4*norm.sf(abs(pts[i]-pts[j])/(2*s))
    return t/16
R['ub']={'Balanced equal':[ub(A_bal,s) for s in snr],'Stage 1':[ub(A_s1,s) for s in snr],'Stage 1+2':[ub(A_s2,s) for s in snr]}
# ---- MC validation
def mc(a,snr_db,n=2_000_000,seed=1):
    r=np.random.default_rng(seed); gs=gsum(a); pts,lab=pts_sm(gs); k=r.integers(0,16,n)
    y=pts[k]+r.normal(0,sigma_for(snr_db),n)
    o=np.argsort(pts); p=pts[o]; l=lab[o]; mid=(p[:-1]+p[1:])/2; dec=np.searchsorted(mid,y)
    err=HD[lab[k],l[dec]].sum()/(4*n); return err
R['mc']={ 'Stage 1+2@25':[mc(A_s2,25),ber_sm(A_s2,25)], 'Stage 1@30':[mc(A_s1,30),ber_sm(A_s1,30)],'Contig@30':[mc(A_con,30),ber_sm(A_con,30)]}
print(R['mc'])
# ---- GSM
R['gsm']={'Balanced equal':[ber_gsm(A_bal,s) for s in snr],'Contiguous':[ber_gsm(A_con,s) for s in snr],'Stage 1':[ber_gsm(A_s1,s) for s in snr],'Stage 1+2':[ber_gsm(A_s2,s) for s in snr]}
print('gsm',{k:[f"{x:.1e}" for x in v[4:]] for k,v in R['gsm'].items()})
# ---- r sensitivity
rs=[1.0,1.1,1.2,1.3,1.4,1.5,1.7,2.0,2.5]
sens={'r':rs,'s1_30':[],'s2_30':[],'s1_25':[],'s2_25':[],'dmin1':[],'dmin2':[]}
for r in rs:
    a1=stage1(geo_w(r)); a2=stage2(a1)
    sens['s1_30'].append(ber_sm(a1,30));sens['s2_30'].append(ber_sm(a2,30));sens['s1_25'].append(ber_sm(a1,25));sens['s2_25'].append(ber_sm(a2,25))
    sens['dmin1'].append(dmin_fast(gsum(a1))*1e12);sens['dmin2'].append(dmin_fast(gsum(a2))*1e12)
R['sens']=sens; print({k:np.round(v,4).tolist() for k,v in sens.items()})
# ---- failures
fr=[0,5,10,20,30]; NT=200; fail={'p':fr}
def ci(x): x=np.array(x); return float(x.mean()),float(1.96*x.std(ddof=1)/np.sqrt(len(x)))
for snrdb in (30,20):
    out={'frozen_bal':[],'frozen_prop':[],'repart':[],'repart_s1':[],'frozen_con':[]}
    for p in fr:
        d={k:[] for k in out}
        for _ in range(NT if p>0 else 1):
            nf=int(round(p/100*N)); alive=np.ones(N,bool)
            if nf: alive[rng.choice(N,nf,replace=False)]=False
            hh=np.where(alive,h,0.0)
            d['frozen_bal'].append(ber_sm(A_bal,snrdb,hh)); d['frozen_prop'].append(ber_sm(A_s2,snrdb,hh)); d['frozen_con'].append(ber_sm(A_con,snrdb,hh))
            hs=h[alive]; a1=stage1(w15,hs); a2=stage2(a1,hs)
            # evaluate on survivors with the common noise reference of the intact panel
            d['repart'].append(ber_sm(a2,snrdb,hs)); d['repart_s1'].append(ber_sm(a1,snrdb,hs))
        for k in out: out[k].append(ci(d[k]) if len(d[k])>1 else (d[k][0],0.0))
    fail[str(snrdb)]=out
R['fail']=fail; print({k:[round(x[0],4) for x in v] for k,v in fail['30'].items()})
# ---- CSI error on partition design
cs=[0,2,5,10,20,30]; csi={'e':cs,'prop':[],'s1':[],'con':[]}
for e in cs:
    d1=[];d2=[]
    for _ in range(NT if e>0 else 1):
        he=h*(1+e/100*rng.standard_normal(N)); he=np.clip(he,1e-14,None)
        a1=stage1(w15,he); a2=stage2(a1,he)
        d1.append(ber_sm(a2,30,h)); d2.append(ber_sm(a1,30,h))
    csi['prop'].append(ci(d1) if len(d1)>1 else (d1[0],0.0)); csi['s1'].append(ci(d2) if len(d2)>1 else (d2[0],0.0))
R['csi']=csi; print({k:[round(x[0],4) for x in v] for k,v in csi.items() if k!='e'})
json.dump(R,open('res.json','w'))
np.save('A_s1.npy',A_s1);np.save('A_s2.npy',A_s2);np.save('A_bal.npy',A_bal);np.save('A_con.npy',A_con)
