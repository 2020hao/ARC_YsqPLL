"""Select completed windows without rewriting fallback from future PLL evidence."""
import numpy as np

def parallel_candidates(good,n,fs):
    """Native and fixed histories remain separate, including before initial lock."""
    native=[dict(start_sample=int(q['start_sample']),stop_sample_exclusive=int(q['stop_sample_exclusive']),kind='pll',segment=int(q['reference_segment']),reference_epoch=int(q.get('reference_epoch',0))) for q in good.to_dict('records')]
    fixed=[dict(start_sample=a,stop_sample_exclusive=a+fs//50,kind='nominal_20ms_fallback',segment=-1,reference_epoch=-1) for a in range(0,n-fs//50+1,fs//50)]
    return native+fixed

def select_available(entries,anchors,valid_at_anchor,segment_at_anchor,epoch_at_anchor,max_age=.04):
    ends=np.array([e['end_seconds'] for e in entries]);available=np.array([e['available_seconds'] for e in entries]);native=np.array([e['kind']=='pll' for e in entries]);segments=np.array([e.get('segment',0) for e in entries]);epochs=np.array([e.get('epoch',0) for e in entries]);out=[]
    for t,v,s,epoch in zip(anchors,valid_at_anchor,segment_at_anchor,epoch_at_anchor):
        eligible=(available<=t+1e-10)&(ends<=t+1e-10)
        good=np.flatnonzero(eligible&native&(segments==s)&(epochs==epoch)&(t-ends<=max_age+1e-10)) if v else np.array([],int)
        ids=good if len(good) else np.flatnonzero(eligible&~native)
        if not len(ids):raise ValueError('No completed causal window at decision anchor')
        out.append(int(ids[np.argmax(ends[ids])]))
    return np.asarray(out)
