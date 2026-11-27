"""Exact float64 CORE3 reductions, including multiplicities of repeated keys.
The original P13/P14 operator is the numerical reference. No sparse truncation.
"""
import numpy as np
from scipy.sparse import csr_matrix

def prepare_scores(scores, multiplicity):
    x=np.asarray(scores,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Scores no finitos/2D.')
    m=np.asarray(multiplicity,dtype=np.float64)
    if m.shape!=(x.shape[1],) or not (m>=1).all() or not np.equal(m,np.floor(m)).all():
        raise ValueError('Multiplicidades enteras positivas requeridas.')
    order=np.argsort(-x,axis=1,kind='stable')
    s=np.take_along_axis(x,order,axis=1);s=s-s[:,0,None]
    return s,order,m[order]

def prefixes(s,m):
    # W, S1,...,S4. Scores are shifted by their maximum before taking powers.
    return [np.cumsum(m,axis=1)]+[np.cumsum(m*s**j,axis=1) for j in range(1,5)]

def threshold(s,pref,gain,network):
    if gain<=0 or not np.isfinite(gain):raise ValueError('Gain debe ser positivo.')
    target=4. if network==1 else 1.
    W,S1,S2=pref[:3];q,N=s.shape;row=np.arange(q);lo=np.zeros(q,dtype=int);hi=np.full(q,N,dtype=int)
    # F(s[k]) is monotone in k; the last F(s[k]) <= target is active.
    while np.any(hi-lo>1):
        mid=(lo+hi)//2;ix=np.minimum(mid,N-1);v=s[row,ix]
        f=gain*gain*np.maximum(S2[row,ix]-2*v*S1[row,ix]+v*v*W[row,ix],0)
        yes=f<=target
        lo=np.where((hi-lo>1)&yes,mid,lo);hi=np.where((hi-lo>1)&~yes,mid,hi)
    k=lo;w=W[row,k];a=S1[row,k];b=S2[row,k]
    disc=np.maximum(w*target-gain*gain*np.maximum(w*b-a*a,0),0)
    tau=(gain*a-np.sqrt(disc))/w
    return tau,k

def weights(s,m,pref,gain,network):
    if network==0:
        v=np.exp(gain*s);group=v*m;return group/group.sum(axis=1,keepdims=True)
    tau,k=threshold(s,pref,gain,network)
    v=np.maximum(gain*s-tau[:,None],0)
    group=m*(v*v if network==1 else v)
    den=group.sum(axis=1,keepdims=True)
    if not np.isfinite(group).all() or not (den>0).all():raise FloatingPointError('Pesos CORE3 inválidos.')
    return group/den

def reference_weights(scores,m,gain,network):
    x=np.asarray(scores,float)*gain;x=x-x.max(axis=1,keepdims=True)
    if network==0:
        v=np.exp(x)*m;return v/v.sum(axis=1,keepdims=True)
    lo=np.full(len(x),-2. if network==1 else -1.);hi=np.zeros(len(x))
    for _ in range(100):
        mid=(lo+hi)/2;v=np.maximum(x-mid[:,None],0)
        f=(m*(v/2)**2).sum(1) if network==1 else (m*v*v).sum(1)
        lo=np.where(f>1,mid,lo);hi=np.where(f>1,hi,mid)
        if np.max(hi-lo)<=1e-12:break
    v=np.maximum(x-((lo+hi)/2)[:,None],0)
    v=m*(v*v if network==1 else v)
    return v/v.sum(1,keepdims=True)

def fraction(s,m,pref,gain,network):
    """N_eff / N counts repeated patterns individually, not compressed groups."""
    N=pref[0][:,-1]
    if network==0:
        v=np.exp(gain*s);return np.square((m*v).sum(1))/(N*(m*v*v).sum(1))
    tau,k=threshold(s,pref,gain,network);row=np.arange(len(s))
    W,A,B,C,D=[x[row,k] for x in pref]
    t2=gain*gain*B-2*gain*tau*A+tau*tau*W
    if network==2:
        t1=gain*A-tau*W;result=t1*t1/(N*t2)
    else:
        t4=gain**4*D-4*gain**3*tau*C+6*gain*gain*tau*tau*B-4*gain*tau**3*A+tau**4*W
        result=t2*t2/(N*t4)
    bad=(~np.isfinite(result))|(result<1/N-1e-10)|(result>1+1e-10)
    if bad.any():
        # Cancellation fallback evaluates the same exact weights, never a new model.
        w=weights(s[bad],m[bad],[x[bad] for x in pref],gain,network)
        result[bad]=1/(N[bad]*(w*w/m[bad]).sum(1))
    if not np.isfinite(result).all() or not (result>0).all():raise FloatingPointError('N_eff inválido.')
    return result

def recall_sorted(s,order,m,pref,gain,network,values):
    w=weights(s,m,pref,gain,network)
    nnz=np.count_nonzero(w)
    if network and nnz<.10*w.size:
        rr,cc=np.nonzero(w);matrix=csr_matrix((w[rr,cc],(rr,order[rr,cc])),shape=w.shape)
        return np.asarray(matrix@values)
    original=np.empty_like(w);np.put_along_axis(original,order,w,axis=1)
    return original@values
