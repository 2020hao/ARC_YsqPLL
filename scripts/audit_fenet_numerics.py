"""Training-only VMD sensitivity: characterize, never silently replace baseline inputs."""
import os
import numpy as np
import pandas as pd
import torch
import complete_paper_baselines as base
from arc_model.representations.mbwvd_torch import transform

OUT=base.ROOT/'outputs/grouped_baseline_standard_20261029/fenet_numerics'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3']
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError(OUT)
    cases=[];coverage=[]
    for ds in base.DATASETS:
        table,tr,_=base.indices(ds,0);keep=np.load(base.COHORT/f'{ds}_indices.npy')
        d=base.FIXED/ds/'fixed20';diag=np.load(d/'vmd_diagnostics.npz')
        coverage.append(dict(dataset=ds,N=len(keep),converged=int(diag['converged'][keep].sum()),
            nonconverged=int((~diag['converged'][keep]).sum()),iteration_cap_reached=int((diag['iterations'][keep]>=500).sum())))
        ids=[]
        for _,g in table.iloc[tr].groupby(['condition_key','label'],sort=True):ids.append(int(g.index[len(g)//2]))
        cases.append(dict(dataset=ds,indices=ids,source=base.sha(d/'native.npy')))
    spec=dict(status='frozen_before_transform',selection='fold0 TRAIN only; midpoint percondition/class, no test-score selection',
        max_iter=[500,2000],tol=1e-7,alpha=2000,cases=cases,
        use='numerical sensitivity only; no automatic input/weight/score replacement',gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    base.save(OUT/'protocol.json',spec);rows=[]
    for c in cases:
        ds=c['dataset'];d=base.FIXED/ds/'fixed20';native=np.load(d/'native.npy',mmap_mode='r');old=np.load(d/'mbwvd.npy',mmap_mode='r')
        for start in range(0,len(c['indices']),8):
            ids=np.array(c['indices'][start:start+8]);x=np.array(native[ids]);a,ca,ia,va=transform(x,max_iter=500)
            np.testing.assert_allclose(a,old[ids],atol=1e-7,rtol=2e-4)
            b,cb,ib,vb=transform(x,max_iter=2000)
            for j,ix in enumerate(ids):
                u=a[j].ravel().astype(float);v=b[j].ravel().astype(float)
                def norm(z):return (z-z.min())/max(z.max()-z.min(),1e-20)
                un,vn=norm(u),norm(v)
                rows.append(dict(dataset=ds,index=int(ix),iter500=int(ia[j]),iter2000=int(ib[j]),
                    converged500=bool(va[j]),converged2000=bool(vb[j]),
                    raw_image_relative_L2=float(np.linalg.norm(u-v)/max(np.linalg.norm(u),1e-20)),
                    normalized_image_RMSE=float(np.sqrt(np.mean((un-vn)**2))),
                    normalized_image_cosine=float(un@vn/max(np.linalg.norm(un)*np.linalg.norm(vn),1e-20)),
                    centers500=ca[j].tolist(),centers2000=cb[j].tolist()))
            print('AUDIT',ds,start+len(ids),len(c['indices']),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'sensitivity.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'full_cohort_convergence.csv',index=False)
    summary=frame.groupby('dataset').agg(samples=('index','size'),converged500=('converged500','mean'),
        converged2000=('converged2000','mean'),median_relL2=('raw_image_relative_L2','median'),
        median_normalized_RMSE=('normalized_image_RMSE','median'),max_normalized_RMSE=('normalized_image_RMSE','max'),
        median_cosine=('normalized_image_cosine','median'),min_cosine=('normalized_image_cosine','min')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    center_rows=[]
    for ds in base.DATASETS:
        d=base.FIXED/ds/'fixed20';keep=np.load(base.COHORT/f'{ds}_indices.npy')
        diag=np.load(d/'vmd_diagnostics.npz');table=pd.read_csv(d/'cycles.csv')
        centers=np.sort(diag['centers'][keep],axis=1);labels=table.iloc[keep].label.to_numpy()
        for label in ['all',0,1]:
            c=centers if label=='all' else centers[labels==label]
            center_rows.append(dict(dataset=ds,label=label,N=len(c),low_median=float(np.median(c[:,0])),
                middle_median=float(np.median(c[:,1])),high_median=float(np.median(c[:,2])),
                middle_in_15_29k=float(((c[:,1]>=15000)&(c[:,1]<=29000)).mean()),
                high_in_98_112k=float(((c[:,2]>=98000)&(c[:,2]<=112000)).mean())))
    pd.DataFrame(center_rows).to_csv(OUT/'mode_center_audit.csv',index=False)
    base.save(OUT/'COMPLETE.json',dict(status='complete',samples=len(frame),training_only=True,unchanged_baseline_inputs=True,
        protocol_sha256=base.sha(OUT/'protocol.json')))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
