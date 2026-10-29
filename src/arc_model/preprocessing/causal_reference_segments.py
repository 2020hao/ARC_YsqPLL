"""Dense run identities shared by candidate generation and online selection."""
import numpy as np

def reference_segment_ids(valid,epoch):
    mask=np.asarray(valid,bool);epoch=np.asarray(epoch)
    starts=mask&(~np.r_[False,mask[:-1]]|np.r_[True,epoch[1:]!=epoch[:-1]])
    return np.cumsum(starts)-1
