"""FP32 LIF scan with explicit gradient matching detached-reset surrogate BPTT."""
from typing import List
import torch


@torch.jit.script
def forward_scan(current:torch.Tensor,valid:torch.Tensor,leak:torch.Tensor,threshold:torch.Tensor,spiking:bool):
    b,t,c,p=current.shape
    previous=torch.zeros_like(current[:,0]);pre=torch.empty_like(current);spikes=torch.empty_like(current)
    for i in range(t):
        mask=valid[:,i].view(b,1,1)
        u=(previous*leak+current[:,i])*mask
        s=((u>=threshold).to(current.dtype) if spiking else torch.sigmoid(3*(u-threshold)))*mask
        previous=(u-threshold*s)*mask
        pre[:,i]=u;spikes[:,i]=s
    return spikes,pre


@torch.jit.script
def backward_scan(gradient:torch.Tensor,pre:torch.Tensor,spikes:torch.Tensor,valid:torch.Tensor,leak:torch.Tensor,threshold:torch.Tensor,spiking:bool):
    b,t,c,p=pre.shape
    future=torch.zeros_like(pre[:,0]);gcurrent=torch.empty_like(pre)
    gleak=torch.zeros_like(leak);gthreshold=torch.zeros_like(threshold)
    for j in range(t):
        i=t-1-j;mask=valid[:,i].view(b,1,1);u=pre[:,i];s=spikes[:,i]
        prime=1/(1+5*(u-threshold).abs()).square() if spiking else 3*s*(1-s)
        local=gradient[:,i]*mask*prime
        gu=future*mask+local
        gthreshold+=(-future*mask*s-local).sum(dim=(0,2),keepdim=True)
        if i>0:
            previous=(pre[:,i-1]-threshold*spikes[:,i-1])*valid[:,i-1].view(b,1,1)
            gleak+=(gu*mask*previous).sum(dim=(0,2),keepdim=True)
        gcurrent[:,i]=gu*mask;future=gu*mask*leak
    return gcurrent,gleak,gthreshold


class LIFScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,current,valid,leak,threshold,spiking):
        outputs,pre=forward_scan(current,valid,leak,threshold,spiking)
        ctx.save_for_backward(pre,outputs,valid,leak,threshold);ctx.spiking=spiking
        return outputs
    @staticmethod
    def backward(ctx,gradient):
        pre,outputs,valid,leak,threshold=ctx.saved_tensors
        gi,gl,gt=backward_scan(gradient,pre,outputs,valid,leak,threshold,ctx.spiking)
        return gi,None,gl,gt,None
