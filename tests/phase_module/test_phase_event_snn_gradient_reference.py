import sys,torch
sys.path.insert(0,'src')
from arc_model.implementations.phase_event_snn.neurons import LIFScan
from arc_model.implementations.phase_event_snn.network import Spike
for spiking in (True,False):
 torch.manual_seed(5)
 x=torch.randn(3,7,4,5,requires_grad=True);valid=torch.ones(3,7);valid[:,:2]=0;valid[1,4]=0
 leak=torch.full((1,4,1),.8,requires_grad=True);threshold=torch.full((1,4,1),.5,requires_grad=True)
 u=torch.zeros_like(x[:,0]);out=[]
 for t in range(x.shape[1]):
  m=valid[:,t,None,None];u=(leak*u+x[:,t])*m;s=(Spike.apply(u-threshold) if spiking else torch.sigmoid(3*(u-threshold)))*m
  u=(u-threshold*s.detach())*m;out.append(s)
 ref=torch.stack(out,1);g=torch.randn_like(ref);ref.backward(g);expected=[z.grad.clone() for z in (x,leak,threshold)]
 for z in (x,leak,threshold):z.grad=None
 actual=LIFScan.apply(x,valid,leak,threshold,spiking);torch.testing.assert_close(actual,ref,rtol=0,atol=0)
 actual.backward(g)
 for a,b in zip(expected,(x.grad,leak.grad,threshold.grad)):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
 print('PASS',spiking)
