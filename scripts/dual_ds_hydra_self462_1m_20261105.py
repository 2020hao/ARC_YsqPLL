"""Solve the same frozen HYDRA hinge LP via its exactly equivalent smaller-row dual."""
import os,time,argparse,subprocess,json,signal
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeResult
import _bootstrap as boot
from run_self462_1m_baselines_20261105 import OUT,PYTHON,hydra_fit,sha,save
CFG=boot.ROOT/'configs/modules/self462_1m_hydra_dual_ds_20261105.yaml'
CURRENT_FOLD=None

def solve(c,**kw):
 A=kw['A_ub'];n=A.shape[0];d=(len(c)-n-1)//2;assert len(c)==2*d+1+n
 lam=c[0];np.testing.assert_array_equal(c[:2*d],np.full(2*d,lam));np.testing.assert_allclose(c[2*d+1:],1/n);assert c[2*d]==0
 a=A[:,:d];assert (A[:,d:2*d]+a).nnz==0;sign=-A[:,2*d].toarray().ravel();assert set(sign)=={-1.,1.};tail=A[:,2*d+1:];assert tail.shape==(n,n) and tail.nnz==n and (tail+sparse.eye(n)).nnz==0
 tick=time.time();r=linprog(-np.ones(n),A_ub=sparse.vstack([-a.T,a.T],format='csc'),b_ub=np.full(2*d,lam),A_eq=sparse.csr_matrix(sign[None,:]),b_eq=[0.],bounds=(0,1/n),method='highs-ds',options=dict(time_limit=1800,primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8,disp=True))
 if not r.success:return r
 w=-r.ineqlin.marginals[:d]+r.ineqlin.marginals[d:];b=-r.eqlin.marginals[0];margin=-(a@w)+sign*b;xi=np.maximum(1-margin,0);primal=np.r_[np.maximum(w,0),np.maximum(-w,0),b,xi];objective=float(c@primal);dual=-float(r.fun);gap=abs(objective-dual);slack=-np.ones(n)-A@primal;violation=max(0.,float(-slack.min()));assert gap<1e-6 and violation<1e-6
 cert=dict(status='optimal',method='highs-ipm on exactly equivalent dual LP',seconds=time.time()-tick,objective=objective,dual_objective=dual,absolute_duality_gap=gap,max_constraint_violation=violation,dual_equality_residual=float(abs(sign@r.x)),complementarity=float(abs(r.x*slack).max()),iterations=int(r.nit),crossover_iterations=int(getattr(r,'crossover_nit',0)),original_primal_shape=list(A.shape),dual_shape=[2*d,n],model_objective_unchanged=True)
 if CURRENT_FOLD is not None:save(OUT/f'hydra_ipm_f{CURRENT_FOLD}.json',cert)
 return OptimizeResult(success=True,status=0,message='Optimal (HiGHS equivalent-dual simplex; original primal certificate passed)',x=primal,fun=objective,nit=r.nit,ineqlin=OptimizeResult(residual=slack,marginals=-r.x))

def parity():
 rng=np.random.default_rng(32);rows=[]
 for n,d in [(200,16),(1000,64)]:
  X=rng.normal(size=(n,d));sg=2*rng.integers(0,2,n)-1;a=sparse.csr_matrix(-sg[:,None]*X);A=sparse.hstack([a,-a,sparse.csr_matrix(-sg[:,None]),-sparse.eye(n)],format='csr');c=np.r_[np.full(2*d,.001),0.,np.full(n,1/n)];kw=dict(A_ub=A,b_ub=-np.ones(n),bounds=[(0,None)]*(2*d)+[(None,None)]+[(0,None)]*n,method='highs');p=linprog(c,**kw);q=solve(c,**kw);assert p.success and q.success and abs(p.fun-q.fun)<1e-7;rows.append(dict(n=n,d=d,primal_objective=p.fun,dual_recovered_primal_objective=q.fun,error=abs(p.fun-q.fun)))
 save(OUT/'HYDRA_DUAL_DS_MATH_PARITY.json',dict(status='passed',rows=rows))

def worker(part):
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 from arc_model.implementations.baseline_hydra import features
 features.linprog=solve
 global CURRENT_FOLD
 for f in range(part,4,2):CURRENT_FOLD=f;print('DUAL_START',f,flush=True);hydra_fit(f)

def run_study(cfg,execute=False):
 if not execute:print('same HYDRA objective; exact dual simplex, 4fold');return
 from check_baseline_release import verify
 verify();parity();save(OUT/'HYDRA_DUAL_DS_EXECUTION_OVERRIDE.json',dict(reason='Dual-IPM f0/f1 returned HiGHS Status4 solve error; retry same exact dual with dual simplex, no classifier change',old_CPU_pids=[866751,866752],unchanged='frozen Model/feature/scaler/all training rows/lambda/.001/mean hinge L1/margin0',algorithm='HiGHS-dual-simplex explicit dual,1800s',hashes={str(p):sha(p) for p in [CFG,Path(__file__),OUT/'HYDRA_SOLVER_EXECUTION_OVERRIDE.json',OUT/'HYDRA_DUAL_DS_MATH_PARITY.json',OUT/'hydra_dual_cpu0.log',OUT/'hydra_dual_cpu1.log',OUT/'HYDRA_DUAL_EXECUTION_OVERRIDE.json']}))
 children=[];logs=[]
 for part in [0,1]:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OPENBLAS_NUM_THREADS']='2';env['OMP_NUM_THREADS']='2';log=open(OUT/f'hydra_dual_ds_cpu{part}.log','a');logs.append(log);children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--worker',str(part)],env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(p.poll() is None for p in children):time.sleep(15)
 for log in logs:log.close()
 assert all(p.returncode==0 for p in children),[p.returncode for p in children]
 save(OUT/'HYDRA_IPM_COMPLETE.json',dict(status='complete',folds=4,objective_unchanged=True,solver='explicit equivalent dual HiGHS-dual-simplex',default900s_failures_preserved=True,primal_IPM_interruptions_preserved=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args();worker(a.worker)
