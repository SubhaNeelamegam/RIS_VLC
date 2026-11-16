import numpy as np, itertools
from scipy.stats import norm
from geom import gains
h_all,ok=gains(); idx=np.where(ok)[0]; h=h_all[idx]; N=len(h); G=4
htot=h.sum()
PAM=np.array([-3,-1,1,3])/np.sqrt(5)
PAMgray=[0b00,0b01,0b11,0b10]   # gray label for levels -3,-1,1,3
def popc(x): return bin(x).count('1')
HD=np.array([[popc(a^b) for b in range(16)] for a in range(16)])
def pts_sm(gs):
    pts=[];lab=[]
    for g in range(4):
        for l in range(4):
            pts.append(gs[g]*PAM[l]); lab.append((g<<2)|PAMgray[l])
    return np.array(pts),np.array(lab)
def ber_line(pts,lab,sigma,nb=4):
    """exact ML BER for scalar constellation with equal priors"""
    o=np.argsort(pts); p=pts[o]; l=lab[o]; K=len(p)
    mid=(p[:-1]+p[1:])/2
    edges=np.concatenate([[-np.inf],mid,[np.inf]])
    tot=0.0
    for k in range(K):
        cdf=norm.cdf((edges-p[k])/sigma); pr=np.diff(cdf)   # prob decide j
        tot+=np.sum(pr*HD[l[k],l])
    return tot/(K*nb)
def sigma_for(snr_db,ref=None):
    ref=(htot/4) if ref is None else ref
    return ref/np.sqrt(10**(snr_db/10))      # E[s^2]=1
def gsum(assign,hh=None):
    hh=h if hh is None else hh
    return np.array([hh[assign==g].sum() for g in range(G)])
# --- partitions
from geom import iy,iz
iz_a=iz[idx]; iy_a=iy[idx]
def balanced_equal(hh=None):
    """equal-size groups with balanced gain sums (snake / LPT with size cap)"""
    hh=h if hh is None else hh; n=len(hh)
    cap=int(np.ceil(n/G)); a=-np.ones(n,int); s=np.zeros(G); c=np.zeros(G,int)
    for i in np.argsort(-hh):
        cand=[g for g in range(G) if c[g]<cap]; g=min(cand,key=lambda x:s[x]); a[i]=g; s[g]+=hh[i]; c[g]+=1
    return a
def random_equal(rng,n=None):
    n=N if n is None else n; a=np.arange(n)%G; rng.shuffle(a); return a
def unequal_size(rng,w,n=None):
    n=N if n is None else n; w=np.array(w,float); cnt=np.floor(w/w.sum()*n).astype(int)
    cnt[0]+=n-cnt.sum(); a=np.repeat(np.arange(G),cnt); rng.shuffle(a); return a
def geo_w(r): w=np.array([r**3,r**2,r,1.0]); return w/w.sum()
def stage1(w,hh=None):
    hh=h if hh is None else hh; tgt=w*hh.sum(); a=-np.ones(len(hh),int); s=np.zeros(G)
    for i in np.argsort(-hh):
        g=int(np.argmax(tgt-s)); a[i]=g; s[g]+=hh[i]
    return a
def dmin_w(gs):
    pts,lab=pts_sm(gs); best=np.inf
    for i in range(16):
        for j in range(i+1,16):
            d=abs(pts[i]-pts[j])/ (HD[lab[i],lab[j]]); best=min(best,d)
    return best
_I,_J=np.triu_indices(16,1)
def dmin_fast(gs):
    pts,lab=pts_sm(gs); d=np.abs(pts[_I]-pts[_J])/HD[lab[_I],lab[_J]]; return d.min()
def stage2(a,hh=None,maxit=200):
    hh=h if hh is None else hh; a=a.copy(); gs=gsum(a,hh); cur=dmin_fast(gs)
    for _ in range(maxit):
        best=(cur,None,None)
        for i in range(len(hh)):
            g0=a[i]
            for g1 in range(G):
                if g1==g0: continue
                gs2=gs.copy(); gs2[g0]-=hh[i]; gs2[g1]+=hh[i]
                d=dmin_fast(gs2)
                if d>best[0]+1e-30: best=(d,i,g1)
        if best[1] is None: break
        i,g1=best[1],best[2]; gs[a[i]]-=hh[i]; gs[g1]+=hh[i]; a[i]=g1; cur=best[0]
    return a
def ber_sm(a,snr_db,hh_true=None):
    hh=h if hh_true is None else hh_true
    gs=gsum(a,hh); pts,lab=pts_sm(gs); return ber_line(pts,lab,sigma_for(snr_db))
# 16-PAM all elements, gray
def gray(n): return n^(n>>1)
def ber_16pam(snr_db,peak=False):
    lev=(np.arange(16)*2-15).astype(float)
    lev=lev/np.sqrt(np.mean(lev**2)) if not peak else lev/ (15/ (3/np.sqrt(5)))  # peak-limited: same peak as 4-PAM(=3/sqrt5)
    pts=htot*lev; lab=np.array([gray(k) for k in range(16)])
    return ber_line(pts,lab,sigma_for(snr_db))
# GSM: two active groups, choose 4 of 6 pairs
pairs=list(itertools.combinations(range(4),2))
def ber_gsm(a,snr_db,hh=None):
    hh=h if hh is None else hh; gs=gsum(a,hh)
    sums=np.array([gs[i]+gs[j] for i,j in pairs])
    best=None
    for comb in itertools.combinations(range(6),4):
        sv=np.sort(sums[list(comb)]); 
        # composite min dist
        P=[];L=[]
        for q,pi_ in enumerate(comb):
            for l in range(4): P.append(sums[pi_]*PAM[l]); L.append((q<<2)|PAMgray[l])
        P=np.array(P);L=np.array(L)
        d=np.abs(P[_I]-P[_J])/HD[L[_I],L[_J]]
        if best is None or d.min()>best[0]: best=(d.min(),P,L)
    return ber_line(best[1],best[2],sigma_for(snr_db))
