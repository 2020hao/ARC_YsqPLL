"""Load an enhanced classifier and score native-valued phase profiles."""
from pathlib import Path
import torch
from .network import DynamicStateNet


def load_candidate_checkpoint(checkpoint,device='cpu'):
    model=DynamicStateNet('maxpool_dynamic')
    model.load_state_dict(torch.load(Path(checkpoint),map_location='cpu',weights_only=True))
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_phase_profiles(model,native_profiles,history_mask):
    """Score [N,9,5,64] phase profiles; this interface is after raw RCR/HF.

    Native here means original-valued LF/log10-power/delta profiles, before
    training-fold standardization, not a raw MHz current waveform. All input
    and summary normalization state resides in the checkpoint buffers.
    """
    target=next(model.parameters())
    values=torch.as_tensor(native_profiles,dtype=target.dtype,device=target.device)
    mask=torch.as_tensor(history_mask,dtype=target.dtype,device=target.device)
    if values.ndim!=4 or values.shape[1:]!=(9,5,64) or mask.shape!=values.shape[:2]:
        raise ValueError('Expected profiles[N,9,5,64] and mask[N,9]')
    x=(values-model.input_mean[None,None,:,None])/model.input_std[None,None,:,None]
    return model(x,mask).sigmoid()
