"""Data-host read-only native CH0 -> causal50k and12.8k caches."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,csv,json,time,struct,hashlib
import numpy as np
from scipy.signal import upfirdn,firwin,kaiserord,freqz

def filters():
 n,beta=kaiserord(80,(6400-5000)/(50000*32/2));n+=1-n%2
 return firwin(801,20000,fs=2000000,window=('kaiser',7.5)),firwin(n,5700,fs=1600000,window=('kaiser',beta))*32

def one(job):
 row,outdir=job;dest=Path(outdir)/row['id'];dest.mkdir(parents=True,exist_ok=True);done=dest/'COMPLETE.json'
 if done.exists():return json.loads(done.read_text())
 path=Path(row['bin_114']);start=time.time()
 with path.open('rb') as f:header=f.read(56)
 dev,ts,ch,fs,n=struct.unpack('<32sQIIQ',header);assert dev.rstrip(b'\0')==b'NET9770_114' and ch==8 and fs==2000000
 assert path.stat().st_size==56+n*ch*8
 raw=np.memmap(path,dtype='<f8',mode='r',offset=56,shape=(n,ch))[:,0].copy();assert np.isfinite(raw).all();h=hashlib.sha256(raw.tobytes()).hexdigest();f1,f2=filters()
 low50=upfirdn(f1,raw,down=40)[:(n+39)//40];low128=upfirdn(f2,low50,up=32,down=125)[:(len(low50)*32+124)//125]
 assert np.isfinite(low50).all() and np.isfinite(low128).all()
 np.savez_compressed(dest/'signals.npz',low50=low50,low128=low128)
 receipt=dict(id=row['id'],CH0_sha256=h,header_sha256=hashlib.sha256(header).hexdigest(),source_size=path.stat().st_size,native_samples=n,low50_samples=len(low50),low128_samples=len(low128),native_duration_s=n/fs,seconds=time.time()-start)
 done.write_text(json.dumps(receipt));return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader(open(a.manifest,encoding='utf-8-sig')))
 f1,f2=filters();np.savez(out/'resampling_filters.npz',first=f1,second=f2)
 _,h=freqz(f2/32,worN=131072,fs=1600000);freq=np.arange(len(h))*1600000/(2*len(h));db=20*np.log10(np.maximum(abs(h),1e-30))
 report=dict(second_taps=len(f2),second_pass_ripple_dB=float(np.ptp(db[freq<=5000])),second_stop_max_dB=float(db[freq>=6400].max()),first_delay_ms=400/2000000*1000,second_delay_ms=(len(f2)-1)/2/1600000*1000,causal_no_advance=True)
 assert report['second_stop_max_dB'] < -75 and report['second_pass_ripple_dB']<.01
 (out/'resampling_spec.json').write_text(json.dumps(report,indent=2));receipts=[]
 with ProcessPoolExecutor(a.workers) as pool:
  jobs=[pool.submit(one,(r,str(out/'records'))) for r in rows]
  for f in as_completed(jobs):
   rec=f.result();receipts.append(rec);(out/'progress.json').write_text(json.dumps(dict(done=len(receipts),total=len(rows),last=rec)));print('SIGNALS',len(receipts),round(rec['seconds'],2),flush=True)
 (out/'PREPARED.json').write_text(json.dumps(dict(status='complete',records=receipts,resampling=report)))
if __name__=='__main__':main()
