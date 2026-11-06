"""Same3945-parameter hybrid, with deterministic coding of standardized inputs."""
import torch
from ._source_direct_efficiency_phase_map_hybrid import PhaseMapHybrid


def code_inputs(x,coding):
    if coding=='raw':return x
    clipped=x.clamp(-3.,3.)
    if coding=='clip':return clipped
    if coding!='sparse4':raise ValueError('Unknown input coding')
    magnitude=clipped.abs();amplitude=((magnitude-.25)/2.75).clamp(0,1)
    # Equivalent amplitude reconstruction of the existing4step signed TTFS code.
    latency=torch.round(3*(1-amplitude))
    decoded=.25+(1-latency/3)*2.75
    return clipped.sign()*decoded*(magnitude>=.25).to(x.dtype)


class InputCodedPhaseMap(PhaseMapHybrid):
    def __init__(self,coding='raw'):
        super().__init__();self.input_coding=coding
    def forward(self,x,mask,return_activity=False,return_details=False):
        return super().forward(code_inputs(x,self.input_coding),mask,return_activity,return_details)
