"""Weighted-operation accounting; sparse equivalents are not measured energy."""
import torch


def operations(details,encoding,spiking,original=None):
    e=details['encoded'];p1=details['pool1'];p2=details['pool2'];s3=details['layer3'];valid=details['valid']
    sums=lambda t:t.count_nonzero(dim=tuple(range(1,t.ndim))).to(torch.float64)
    first=sums(e)*80
    second=sums(p1)*72
    third=sums(p2)*16
    final=sums(s3[:,-4:])
    zeros=torch.zeros_like(first)
    ac=(first if encoding!='direct' else zeros)+(second+third+final if spiking else zeros)
    mac=(first if encoding=='direct' else zeros)+(second+third+final if not spiking else zeros)
    batch=e.shape[0];valid_steps=valid.sum(1).double()
    current=e[:,-4:];dynamic_indices=torch.tensor([3,4,11,12],device=e.device)
    dynamic=sums(current.index_select(2,dynamic_indices))
    result=dict(input_nonzeros=sums(e),current_cycle_input_nonzeros=sums(current),
        current_cycle_dynamic_nonzeros=dynamic,effective_weighted_AC=ac,effective_weighted_MAC=mac,
        dense_weighted_MAC=torch.full((batch,),4497472.,device=e.device,dtype=torch.float64),
        valid_neuron_updates=valid_steps*1808,dense_neuron_updates=torch.full((batch,),65088.,device=e.device,dtype=torch.float64),
        normalization_scalar_values=valid_steps*1808,real_leak_state_MAC=valid_steps*1808)
    for i,key in enumerate(('layer1','layer2','layer3'),1):
        s=details[key];size=s[0,0].numel()
        result[f'layer{i}_nonzeros']=sums(s)
        result[f'layer{i}_activation_rate']=s.sum(tuple(range(1,s.ndim))).double()/(valid_steps*size).clamp_min(1)
    if original is not None:
        shaped=e.reshape(e.shape[0],9,4,16,64)
        count=shaped.sum(2)
        if encoding=='ttfs':
            step=torch.arange(4,device=e.device)[None,None,:,None,None]
            amplitude=1-(shaped*step).sum(2)/3;active=count>0
        elif encoding=='rate':
            amplitude=((count+.5)/4).clamp_max(1);active=count>0
        else:amplitude=shaped.mean(2);active=amplitude>0
        magnitude=(.25+amplitude*2.75)*active
        decoded=magnitude[:,:,:8]-magnitude[:,:,8:]
        result['current_encoding_MSE_clipped']=((decoded[:,-1]-original[:,-1].clamp(-3,3))**2).mean((1,2)).double()
        result['current_saturation_MSE']=((original[:,-1]-original[:,-1].clamp(-3,3))**2).mean((1,2)).double()
    return result
