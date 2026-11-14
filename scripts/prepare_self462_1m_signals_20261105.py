"""Read-only data-host conversion to contiguous1MHz CH0; originals unchanged."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,csv,json,time,hashlib
from self462_recording import read_recording
import numpy as np
from scipy.signal import firwin,kaiserord,freqz,upfirdn

def antialias():
 n,beta=kaiserord(80,(500000-400000)/1000000);n+=1-n%2
 return firwin(n,450000,fs=2000000,window=('kaiser',beta))

def one(job, *, recording=None):
 row,outdir=job;d=Path(outdir)/row['id'];d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():return json.loads((d/'COMPLETE.json').read_text())
 start=time.time();p=Path(row['bin_114'])
 recording = read_recording(p) if recording is None else recording
 h, n, raw, source_hash = recording
 y=upfirdn(antialias(),raw,down=2)[:(n+1)//2];assert np.isfinite(y).all();np.save(d/'current1m.npy',y)
 receipt=dict(id=row['id'],source_CH0_sha256=source_hash,source_header_sha256=hashlib.sha256(h).hexdigest(),source_size=p.stat().st_size,source_samples=n,samples1m=len(y),fs=1000000,signal_sha256=hashlib.sha256(y.tobytes()).hexdigest(),seconds=time.time()-start)
 (d/'COMPLETE.json').write_text(json.dumps(receipt));return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 f=antialias();np.save(out/'AA_FIR.npy',f);w,h=freqz(f,worN=131072,fs=2000000);db=20*np.log10(np.maximum(abs(h),1e-30));spec=dict(taps=len(f),pass_ripple_dB=float(np.ptp(db[w<=400000])),stop_max_dB=float(db[w>=500000].max()),delay_ms=(len(f)-1)/2/2000000*1000,causal=True)
 assert spec['pass_ripple_dB']<.01 and spec['stop_max_dB']<-75
 (out/'AA_spec.json').write_text(json.dumps(spec,indent=2));rows=list(csv.DictReader(open(a.manifest,encoding='utf-8-sig')));done=[]
 with ProcessPoolExecutor(a.workers) as pool:
  jobs=[pool.submit(one,(r,str(out/'records'))) for r in rows]
  for q in as_completed(jobs):
   r=q.result();done.append(r);(out/'progress.json').write_text(json.dumps(dict(completed=len(done),total=len(rows))));print('SIGNAL1M',len(done),round(r['seconds'],2),flush=True)
 (out/'PREPARED.json').write_text(json.dumps(dict(status='complete',records=done,spec=spec)))
if __name__=='__main__':main()
