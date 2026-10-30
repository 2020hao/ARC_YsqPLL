"""FP64 VMD equations with CUDA graph replay; research qualification, not frozen code."""
import numpy as np
import torch


class VMDGraph:
    def __init__(self, batch=64, max_iter=2000, alpha=2000., tol=1e-7, steps=25):
        if max_iter % steps:raise ValueError('Graph steps must divide maximum iterations')
        self.batch=batch;self.max_iter=max_iter;self.alpha=alpha;self.tol=tol;self.steps=steps
        self.input=torch.zeros((batch,5000),device='cuda',dtype=torch.float64)
        self.modes=torch.zeros((batch,3,5001),device='cuda',dtype=torch.complex128)
        self.centers=torch.zeros((batch,3),device='cuda',dtype=torch.float64)
        self.spectrum=torch.zeros((batch,5001),device='cuda',dtype=torch.complex128)
        self.freq=torch.fft.rfftfreq(10000,device='cuda',dtype=torch.float64)
        self.done=torch.zeros(batch,device='cuda',dtype=torch.bool)
        self.iterations=torch.full((batch,),max_iter,device='cuda',dtype=torch.int64)
        self.counter=torch.zeros((),device='cuda',dtype=torch.int64)
        self.reset(np.zeros((batch,5000)))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.step()
        torch.cuda.current_stream().wait_stream(stream)
        self.reset(np.zeros((batch,5000)))
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(steps):self.step()
        self.reset(np.zeros((batch,5000)))

    def reset(self,x):
        self.input.copy_(torch.as_tensor(x,dtype=torch.float64,device='cuda'))
        self.scale=self.input.square().mean(1).sqrt().clamp_min(1e-12)
        u=self.input/self.scale[:,None]
        f=torch.cat([u[:,:2500].flip(1),u,u[:,-2500:].flip(1)],1)
        self.spectrum.copy_(torch.fft.rfft(f));self.modes.zero_()
        self.centers.copy_(torch.tensor([50,22000,101000],device='cuda',dtype=torch.float64)[None]/250000)
        self.done.zero_();self.iterations.fill_(self.max_iter);self.counter.zero_()

    def step(self):
        old=self.modes.clone();total=self.modes.sum(1);active=~self.done
        for k in range(3):
            new=(self.spectrum-(total-self.modes[:,k]))/(1+self.alpha*(self.freq[None]-self.centers[:,k,None]).square())
            new=torch.where(active[:,None],new,self.modes[:,k])
            total+=new-self.modes[:,k];self.modes[:,k]=new
            power=new.abs().square();c=(self.freq[None]*power).sum(1)/power.sum(1).clamp_min(1e-30)
            self.centers[:,k]=torch.where(active,c,self.centers[:,k])
        change=(self.modes-old).abs().square().sum((1,2))/10000
        self.counter.add_(1);finished=active&(change<self.tol)
        self.iterations.copy_(torch.where(finished,self.counter,self.iterations));self.done.logical_or_(finished)

    def transform(self,x):
        x=np.asarray(x);n=len(x)
        if not 0<n<=self.batch or x.shape[1:]!=(5000,):raise ValueError('Native250k [N,5000] required')
        padded=np.zeros((self.batch,5000));padded[:n]=x;self.reset(padded)
        for _ in range(self.max_iter//self.steps):
            self.graph.replay()
            if bool(self.done.all()):break
        keep=self.centers.argsort(1)[:,1:]
        selected=torch.gather(self.modes,1,keep[:,:,None].expand(-1,-1,5001)).sum(1)
        high=torch.fft.irfft(selected,n=10000)[:,2500:7500]*self.scale[:,None]
        return wvd(high[:n]),self.centers[:n].cpu().numpy()*250000,self.iterations[:n].cpu().numpy(),self.done[:n].cpu().numpy()


def wvd(high,tile=8):
    n=5000;device=high.device;out=[]
    h=torch.zeros(n,device=device,dtype=torch.float64);h[0]=h[n//2]=1;h[1:n//2]=2
    lag=torch.arange(-6,7,device=device);center=2*torch.arange(n,device=device);a=center[:,None]+lag;b=center[:,None]-lag
    valid=(a>=0)&(b>=0)&(a<2*n)&(b<2*n);a=a.clamp(0,2*n-1);b=b.clamp(0,2*n-1)
    win=torch.hamming_window(13,periodic=False,dtype=torch.float64,device=device)
    pos=torch.linspace(0,128,100,device=device,dtype=torch.float64);left=pos.floor().long();right=pos.ceil().long();fraction=pos-left
    for start in range(0,len(high),tile):
        z=torch.fft.ifft(torch.fft.fft(high[start:start+tile])*h);sp=torch.fft.fft(z)
        ext=torch.zeros((len(z),2*n),device=device,dtype=torch.complex128)
        ext[:,:n//2+1]=sp[:,:n//2+1];ext[:,-(n//2-1):]=sp[:,-(n//2-1):];ext[:,n//2]*=.5;ext[:,2*n-n//2]=ext[:,n//2]
        analytic=torch.fft.ifft(ext)*2;product=analytic[:,a]*analytic[:,b].conj()*valid*win
        buf=torch.zeros((len(z),n,256),device=device,dtype=torch.complex128);buf[:,:,lag%256]=product
        tf=torch.fft.fft(buf,dim=-1).real[:,:,:129]/250000;tf=tf.reshape(-1,100,50,129).mean(2)
        out.append((tf[:,:,left]*(1-fraction)+tf[:,:,right]*fraction).float().cpu().numpy())
    return np.concatenate(out)
