"""Independent reversible physical channel decomposition and training statistics."""
import numpy as np


def decompose(inputs,decomposition=True):
    x=np.asarray(inputs,dtype='float32');out=np.zeros(x.shape[:2]+(8,64),dtype='float32')
    if not decomposition:out[:,:,:5]=x;return out
    rms=np.maximum(np.sqrt(np.mean(x[:,:,0]**2,axis=-1,keepdims=True)),1e-12)
    out[:,:,0]=x[:,:,0]/rms
    maximum=x[:,:,1:3].max(-1,keepdims=True)
    logpower=maximum+np.log10(np.maximum(np.mean(10.**(x[:,:,1:3]-maximum),axis=-1,keepdims=True),1e-20))
    out[:,:,1:3]=x[:,:,1:3]-logpower
    out[:,:,3:5]=x[:,:,3:5]
    out[:,:,5:7]=np.broadcast_to(logpower,x[:,:,1:3].shape)
    out[:,:,7]=np.broadcast_to(np.log10(rms),x[:,:,0].shape)
    return out


def training_statistics(inputs,indices):
    current=inputs[np.asarray(indices),-1]
    return current.mean((0,2),dtype=np.float64).astype('float32'),np.maximum(current.std((0,2),dtype=np.float64),1e-6).astype('float32')


def condition_class_weights(labels,groups,indices):
    indices=np.asarray(indices);result=np.zeros(len(labels),dtype='float32')
    for group in np.unique(groups[indices]):
        for label in (0,1):
            ids=indices[(groups[indices]==group)&(labels[indices]==label)]
            if not len(ids):raise ValueError('Training condition requires both classes')
            result[ids]=1/len(ids)
    result[indices]/=result[indices].mean();return result
