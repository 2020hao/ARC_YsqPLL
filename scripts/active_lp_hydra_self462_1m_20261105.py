"""Certified working-set solver for the SAME sealed L1-hinge objective, fold2."""
import os,time,json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeResult
from sklearn.linear_model import SGDClassifier
import _bootstrap as boot
import run_self462_1m_baselines_20261105 as base
OUT=base.OUT;CFG=boot.ROOT/'configs/modules/self462_1m_hydra_active_20261105.yaml';RECORD=False

def solve(c,**kw):
 A=kw['A_ub'];n=A.shape[0];d=(len(c)-n-1)//2;lam=float(c[0]);a=A[:,:d];sg=-A[:,2*d].toarray().ravel();assert len(c)==2*d+1+n and (A[:,d:2*d]+a).nnz==0;np.testing.assert_allclose(c[:2*d],lam);np.testing.assert_allclose(c[2*d+1:],1/n)
 X=-sg[:,None]*a.toarray();y=(sg>0).astype(int);tick=time.time();warm=SGDClassifier(loss='hinge',penalty='l1',alpha=lam,fit_intercept=True,max_iter=30,tol=None,shuffle=True,random_state=2,average=True);warm.fit(X,y);margin=sg*(X@warm.coef_.ravel()+warm.intercept_[0]);order=np.argsort(abs(margin-1));free=np.zeros(n,bool);free[order[:min(n,max(512,2*d))]]=True;fixed=np.where(margin<1,1.,0.);history=[]
 for iteration in range(24):
  ids=np.flatnonzero(free);fixed[free]=0;constant=X.T@(sg*fixed);rhs=np.r_[np.full(d,n*lam)-constant,np.full(d,n*lam)+constant];eq=-float(sg@fixed);D=sparse.csc_matrix((X[ids]*sg[ids,None]).T);constraints=sparse.vstack([D,-D],format='csc');print('ACTIVE_LP',iteration,'free',len(ids),'fixed1',int(fixed.sum()),'seconds',round(time.time()-tick,2),flush=True)
  r=linprog(-np.ones(len(ids)),A_ub=constraints,b_ub=rhs,A_eq=sparse.csr_matrix(sg[ids][None,:]),b_eq=[eq],bounds=(0,1),method='highs-ds',options=dict(time_limit=1800,primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8,disp=True))
  if not r.success:
   history.append(dict(iteration=iteration,free=len(ids),status=r.message))
   if len(ids)==n:return r
   free[order[:min(n,max(len(ids)*2,len(ids)+1024))]]=True;continue
  w=-r.ineqlin.marginals[:d]+r.ineqlin.marginals[d:];b=-r.eqlin.marginals[0];margin=sg*(X@w+b);xi=np.maximum(1-margin,0);p=np.r_[np.maximum(w,0),np.maximum(-w,0),b,xi];beta=fixed.copy();beta[ids]=r.x;alpha=beta/n;objective=float(c@p);dual=float(beta.mean());gap=objective-dual;slack=-np.ones(n)-A@p;violation=max(0.,float(-slack.min()));corr=X.T@(sg*alpha);dual_violation=max(0.,float(abs(corr).max()-lam),float(abs(sg@alpha)));history.append(dict(iteration=iteration,free=len(ids),objective=objective,dual_objective=dual,gap=gap,dual_violation=dual_violation,seconds=time.time()-tick));print('ACTIVE_CERT',history[-1],flush=True)
  if -1e-7<=gap<1e-6 and violation<1e-6 and dual_violation<1e-7:
   cert=dict(status='optimal',method='exact full-objective dual working-set HiGHS dual simplex; training-only SGD warmstart, no alternative classifier adopted',seconds=time.time()-tick,objective=objective,dual_objective=dual,absolute_duality_gap=abs(gap),max_constraint_violation=violation,dual_equality_residual=float(abs(sg@alpha)),dual_feature_violation=dual_violation,complementarity=float(abs(alpha*slack).max()),iterations=int(r.nit),working_set_iterations=iteration+1,all_training_examples=n,free_variables=len(ids),fixed_examples_accounted_in_constraints_and_objective=n-len(ids),history=history,model_objective_unchanged=True)
   if RECORD:base.save(OUT/'hydra_ipm_f2.json',cert)
   return OptimizeResult(success=True,status=0,message='Optimal (HiGHS dual working set; ALL original examples and primal-dual certificate passed)',x=p,fun=objective,nit=r.nit,ineqlin=OptimizeResult(residual=slack,marginals=-alpha))
  bad=(~free)&(((fixed==0)&(margin<1-1e-8))|((fixed==1)&(margin>1+1e-8)));before=int(free.sum());free[bad]=True
  if free.sum()==before:free[order[:min(n,max(before*2,before+1024))]]=True
 raise RuntimeError('Working-set optimality not established; no result adopted')

def parity():
 rng=np.random.default_rng(482);n,d=3000,16;X=rng.normal(size=(n,d));sg=2*rng.integers(0,2,n)-1;a=sparse.csr_matrix(-sg[:,None]*X);A=sparse.hstack([a,-a,sparse.csr_matrix(-sg[:,None]),-sparse.eye(n)],format='csr');c=np.r_[np.full(2*d,.001),0.,np.full(n,1/n)];kw=dict(A_ub=A,b_ub=-np.ones(n),bounds=[(0,None)]*(2*d)+[(None,None)]+[(0,None)]*n,method='highs');p=linprog(c,**kw);q=solve(c,**kw);assert p.success and q.success and abs(p.fun-q.fun)<1e-6;base.save(OUT/'HYDRA_ACTIVE_MATH_PARITY.json',dict(status='passed',n=n,d=d,full_primal_objective=p.fun,working_set_objective=q.fun,error=abs(p.fun-q.fun)))

def run_study(cfg,execute=False):
 if not execute:print('fold2 same L1-hinge objective, certified full-training working-set LP');return
 from check_baseline_release import verify
 verify();parity();base.save(OUT/'HYDRA_ACTIVE_EXECUTION.json',dict(reason='f2 exact dual DS exceeded1800s; equivalent working-set optimization, not row subsampling or replacement classifier',only_fold=2,features_scaler_model_lambda_objective_threshold_unchanged=True,warmstart='SGD SAME hinge+L1 target, training only; never used for final predictions without exact full-data primal-dual certificate',hashes={str(p):base.sha(p) for p in [CFG,Path(__file__),OUT/'HYDRA_ACTIVE_MATH_PARITY.json',OUT/'hydra_dual_ds_cpu2.log']}))
 from arc_model.implementations.baseline_hydra import features
 global RECORD
 RECORD=True;features.linprog=solve;base.hydra_fit(2)
 for f in range(4):
  cert=json.loads((OUT/f'hydra_ipm_f{f}.json').read_text());assert cert['status']=='optimal' and cert['absolute_duality_gap']<1e-6 and (base.folder('HYDRA_Baseline',f)/'COMPLETE.json').exists()
 base.save(OUT/'HYDRA_IPM_COMPLETE.json',dict(status='complete',folds=4,objective_unchanged=True,solver='HiGHS exact dual simplex; f2 certified working-set acceleration; all failed attempts preserved'))
if __name__=='__main__':raise SystemExit('Use registered suite')
