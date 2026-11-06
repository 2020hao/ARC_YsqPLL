"""Training-only shared sensor-gain and current-phase perturbations."""
import math
import torch


def augment_standardized(inputs, mean, std, generator, log10_gain_halfwidth=.15, phase_halfwidth=2):
    """Preserve physical nine-cycle relations and native distribution summaries.

    One gain and phase shift are shared by all cycles in a sample. Current
    scales by g, log10 band powers shift by2log10(g), and normalized activity
    differences only rotate. This is not feature clipping or quantization.
    """
    b=inputs.shape[0]
    loggain=(torch.rand(b,device=inputs.device,generator=generator)*2-1)*log10_gain_halfwidth
    gain=torch.pow(10.,loggain)
    shifts=torch.randint(-phase_halfwidth,phase_halfwidth+1,(b,),device=inputs.device,generator=generator)
    indices=(torch.arange(64,device=inputs.device)[None,:]-shifts[:,None])%64
    result=torch.gather(inputs,3,indices[:,None,None,:].expand(-1,9,5,-1)).clone()
    result[:,:,0]=( (result[:,:,0]*std[0]+mean[0])*gain[:,None,None]-mean[0] )/std[0]
    result[:,:,1:3]+=2*loggain[:,None,None,None]/std[None,None,1:3,None]
    return result
