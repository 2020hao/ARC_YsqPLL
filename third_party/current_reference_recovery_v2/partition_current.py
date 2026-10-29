"""V4 partition function copied verbatim; only numpy import added.
Recent reliable lengths are NOT reset at segment/epoch changes.
"""
import numpy as np

def partition(good,n,fs):
 cycles=[];cursor=0;recent=[];gap=0
 def fill(stop):
  nonlocal cursor,gap
  length=int(round(np.median(recent[-3:]))) if recent else fs//50;kind='recent_length_fallback' if recent else 'nominal_20ms_fallback';gap+=1
  while cursor<stop:
   end=min(cursor+length,stop);cycles.append(dict(start_sample=cursor,stop_sample_exclusive=end,kind=kind if end-cursor==length else 'partial_gap',segment=-gap,phase_valid=False,native_sample_count=end-cursor,reference_epoch=-1));cursor=end
 for q in good.to_dict('records'):
  a=int(q['start_sample']);b=min(int(q['stop_sample_exclusive']),n)
  if a<cursor or b<=a:continue
  fill(a);cycles.append(dict(start_sample=a,stop_sample_exclusive=b,kind='pll',segment=int(q['reference_segment']),phase_valid=True,native_sample_count=b-a,reference_epoch=int(q.get('reference_epoch',0))));recent.append(b-a);cursor=b
 fill(n)
 assert cycles[0]['start_sample']==0 and cycles[-1]['stop_sample_exclusive']==n
 assert all(a['stop_sample_exclusive']==b['start_sample'] for a,b in zip(cycles,cycles[1:]))
 return cycles
