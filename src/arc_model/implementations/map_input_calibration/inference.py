"""Label-free, single-network event consumer with training-derived cutoff."""
import json
from pathlib import Path
import numpy as np
import torch
from .network import InputCodedPhaseMap
from .deep_network import DeepCodedPhaseMap


class CodedCyclePredictor:
    def __init__(self,run,calibrated=False,device='cpu'):
        run=Path(run);spec=json.loads((run/'spec.json').read_text())
        self.network=(DeepCodedPhaseMap if spec['family']=='deep' else InputCodedPhaseMap)(spec['coding']).to(device).eval()
        self.network.load_state_dict(torch.load(run/'outer/model.pt',map_location=device,weights_only=True))
        norm=np.load(run/'outer/preprocessing.npz');self.mean=norm['mean'][None,:,None];self.std=norm['std'][None,:,None]
        self.cutoff=json.loads((run/'calibration.json').read_text())['threshold'] if calibrated else .5;self.device=device
    def predict(self,event):
        x=np.zeros((9,8,64),dtype='float32');x[:,:5]=event['input']
        x=torch.as_tensor((x-self.mean)/self.std,device=self.device).unsqueeze(0);mask=torch.as_tensor(event['mask'],device=self.device).unsqueeze(0)
        with torch.inference_mode():probability=float(self.network(x,mask).sigmoid().item())
        return dict(arc_probability=probability,decision=int(probability>=self.cutoff),decision_threshold=self.cutoff,coordinate=event['coordinate'],phase_status=event['phase_status'])
