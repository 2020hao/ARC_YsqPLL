"""Batched float64 GPU execution of the registered CPU VMD/MBWVD equations."""
import torch
import numpy as np

def transform(x,max_iter=500,tol=1e-7,alpha=2000,wvd_tile=8):
    x=torch.as_tensor(x,dtype=torch.float64,device='cuda:0');n=x.shape[1];assert n==5000
    scale=x.square().mean(1).sqrt().clamp_min(1e-12);u=x/scale[:,None];f=torch.cat([u[:,:n//2].flip(1),u,u[:,-n//2:].flip(1)],1);spectrum=torch.fft.rfft(f);freq=torch.fft.rfftfreq(2*n,device=x.device,dtype=torch.float64);modes=torch.zeros((len(x),3,len(freq)),device=x.device,dtype=torch.complex128);centers=torch.tensor([50,22000,101000],device=x.device,dtype=torch.float64)[None].repeat(len(x),1)/250000;done=torch.zeros(len(x),dtype=torch.bool,device=x.device);iterations=torch.full((len(x),),max_iter,device=x.device,dtype=torch.int64)
    for it in range(max_iter):
        old=modes.clone();total=modes.sum(1);active=~done
        for k in range(3):
            new=(spectrum-(total-modes[:,k]))/(1+alpha*(freq[None]-centers[:,k,None]).square());new=torch.where(active[:,None],new,modes[:,k]);total+=new-modes[:,k];modes[:,k]=new;power=new.abs().square();c=(freq[None]*power).sum(1)/power.sum(1).clamp_min(1e-30);centers[:,k]=torch.where(active,c,centers[:,k])
        change=(modes-old).abs().square().sum((1,2))/(2*n);finished=active&(change<tol);iterations[finished]=it+1;done|=finished
        if bool(done.all()):break
    keep=centers.argsort(1)[:,1:];selected=torch.gather(modes,1,keep[:,:,None].expand(-1,-1,len(freq))).sum(1);high=torch.fft.irfft(selected,n=2*n)[:,n//2:n//2+n]*scale[:,None];out=[]
    # scipy hilbert + complex Fourier resample(2N), including split Nyquist.
    h=torch.zeros(n,device=x.device,dtype=torch.float64);h[0]=h[n//2]=1;h[1:n//2]=2
    lag=torch.arange(-6,7,device=x.device);center=2*torch.arange(n,device=x.device);a=center[:,None]+lag;b=center[:,None]-lag;valid=(a>=0)&(b>=0)&(a<2*n)&(b<2*n);a=a.clamp(0,2*n-1);b=b.clamp(0,2*n-1);win=torch.hamming_window(13,periodic=False,dtype=torch.float64,device=x.device)
    positions=torch.linspace(0,128,100,device=x.device,dtype=torch.float64);left=positions.floor().long();right=positions.ceil().long();fraction=positions-left
    for start in range(0,len(x),wvd_tile):
        z=torch.fft.ifft(torch.fft.fft(high[start:start+wvd_tile])*h);sp=torch.fft.fft(z);extended=torch.zeros((len(z),2*n),device=x.device,dtype=torch.complex128);extended[:,:n//2+1]=sp[:,:n//2+1];extended[:,-(n//2-1):]=sp[:,-(n//2-1):];extended[:,n//2]*=.5;extended[:,2*n-n//2]=extended[:,n//2];analytic=torch.fft.ifft(extended)*2;product=analytic[:,a]*analytic[:,b].conj()*valid*win;buf=torch.zeros((len(z),n,256),device=x.device,dtype=torch.complex128);buf[:,:,lag%256]=product;tf=torch.fft.fft(buf,dim=-1).real[:,:,:129]/250000;tf=tf.reshape(-1,100,50,129).mean(2);out.append((tf[:,:,left]*(1-fraction)+tf[:,:,right]*fraction).float().cpu().numpy())
    return np.concatenate(out),centers.cpu().numpy()*250000,iterations.cpu().numpy(),done.cpu().numpy()
