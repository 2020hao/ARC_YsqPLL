"""Candidate SGRF: training-domain stability gated residual logit fusion.

This is a new experimental module, not the transform used in legacy results.
Fit requires grouped out-of-fold backbone logits and an independent validation
partition. Validation may select alpha=0; no held-out improvement is guaranteed.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .enhancement import FeatureTransform


def rates(y, score):
    y=np.asarray(y,dtype=int);pred=np.asarray(score)>=0
    recall=float(np.mean(pred[y==1]));fpr=float(np.mean(pred[y==0]))
    return dict(bacc=(recall+1-fpr)/2,recall=recall,fpr=fpr)


class StabilityGatedResidual:
    """Fit-on-training feature weighting and validation-selected residual gate.

    Groups identify matched target/background training conditions. Physical-run
    isolation and OOF provenance are checked by the experiment driver, not by
    the acknowledgement flag here. Inference takes one sample's
    features and backbone log odds, never its paired healthy control.
    """
    def __init__(self,l2=0.01,effect_cap=1.0,min_paired_groups=3,
                 alpha_grid=(0.,.25,.5,1.),recall_tolerance=0.,fpr_tolerance=0.):
        self.l2=float(l2);self.effect_cap=float(effect_cap)
        if (isinstance(min_paired_groups,(bool,np.bool_)) or
                not np.isscalar(min_paired_groups) or
                not np.isfinite(float(min_paired_groups)) or
                float(min_paired_groups)!=int(float(min_paired_groups))):
            raise ValueError('min_paired_groups must be a positive integer')
        self.min_paired_groups=int(min_paired_groups)
        self.alpha_grid=tuple(sorted(set(float(x) for x in alpha_grid)))
        self.recall_tolerance=float(recall_tolerance);self.fpr_tolerance=float(fpr_tolerance)
        if (not np.isfinite([self.l2,self.effect_cap,self.recall_tolerance,self.fpr_tolerance,*self.alpha_grid]).all() or
                not 0<=self.recall_tolerance<=1 or not 0<=self.fpr_tolerance<=1):
            raise ValueError('Hyperparameters must be finite; tolerances lie in [0,1]')
        if self.l2<=0 or self.effect_cap<=0 or self.min_paired_groups<1:
            raise ValueError('Positive regularization/cap and paired-group count required')
        if not self.alpha_grid or self.alpha_grid[0]!=0 or any(x<0 or x>1 for x in self.alpha_grid):
            raise ValueError('Gate grid must include zero and lie in [0,1]')

    @staticmethod
    def _ids(values,n,label):
        values=np.asarray(values,dtype=object)
        if values.shape!=(n,):raise ValueError(label+' must match rows')
        missing={'','none','nan','nat','<na>'}
        if any(v is None or str(v).strip().lower() in missing or
               (isinstance(v,(float,np.floating)) and not np.isfinite(v)) for v in values):
            raise ValueError(label+' cannot contain missing/nonfinite IDs')
        return np.asarray([str(v) for v in values])

    @staticmethod
    def _check(x,y,score,groups):
        x=np.asarray(x,float);y=np.asarray(y);score=np.asarray(score,float)
        if x.ndim!=2 or x.shape[1]==0 or y.shape!=(len(x),) or score.shape!=y.shape:
            raise ValueError('Expected X[n,d] and y/log_odds/groups[n]')
        groups=StabilityGatedResidual._ids(groups,len(x),'Group IDs')
        if set(np.unique(y))!={0,1} or not np.isfinite(x).all() or not np.isfinite(score).all():
            raise ValueError('Both binary labels and finite inputs required')
        return x,y.astype(float),score,groups

    def _transform(self,x):
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            h=self.transformer_.transform(x).astype(float)
        if not np.isfinite(h).all():raise ValueError('Feature transform produced nonfinite values')
        return h

    def fit(self,x,y,backbone_oof_log_odds,groups,*,
            x_val,y_val,backbone_val_log_odds,val_groups,val_record_ids,training_scores_are_oof=False):
        if not training_scores_are_oof:
            raise ValueError('Training logits must come from grouped out-of-fold backbone predictions')
        x,y,z,g=self._check(x,y,backbone_oof_log_odds,groups)
        xv,yv,zv,gv=self._check(x_val,y_val,backbone_val_log_odds,val_groups)
        if x.shape[1]!=xv.shape[1] or set(g)&set(gv):
            raise ValueError('Feature counts must match and train/validation groups must be disjoint')
        ids=self._ids(val_record_ids,len(yv),'Validation record IDs')
        records=[np.flatnonzero(ids==rid) for rid in np.unique(ids)]
        if any(len(set(yv[ii]))!=1 or len(set(gv[ii]))!=1 for ii in records):
            raise ValueError('Each validation record must have one label and one group')
        def record_rates(score):
            return rates(np.array([yv[ii[0]] for ii in records]),np.array([score[ii].mean() for ii in records]))
        self.validation_record_count_=len(records)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            self.transformer_=FeatureTransform().fit(x,y,'enhanced')
        h=self._transform(x)
        hv=self._transform(xv)
        effects=[]
        for group in np.unique(g):
            mask=g==group
            if set(y[mask])=={0.,1.}:
                effects.append(np.median(h[mask&(y==1)],axis=0)-np.median(h[mask&(y==0)],axis=0))
        self.paired_groups_=len(effects)
        if len(effects)<self.min_paired_groups:
            raise ValueError('Too few paired training groups for stability estimation')
        effects=np.asarray(effects)
        # Agreement is insensitive to the choice of global sign. Magnitude is
        # saturated; large single-condition effects do not grow without bound.
        signs=np.where(np.abs(effects)>1e-8,np.sign(effects),0.)
        self.stability_=np.abs(signs.mean(axis=0))*np.median(np.minimum(np.abs(effects)/self.effect_cap,1.),axis=0)
        h*=self.stability_;hv*=self.stability_
        a=np.column_stack([h,np.ones(len(h))]);av=np.column_stack([hv,np.ones(len(hv))])
        unique,inv,counts=np.unique(g,return_inverse=True,return_counts=True)
        weights=1./(len(unique)*counts[inv])
        def objective(w):
            score=z+a@w
            loss=float(np.dot(weights,np.logaddexp(0.,score)-y*score)+self.l2*np.dot(w,w)/2)
            grad=a.T@(weights*(expit(score)-y))+self.l2*w
            return loss,grad
        result=minimize(objective,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
        if not result.success:raise RuntimeError('Residual fitting failed: '+str(result.message))
        self.coef_=result.x;self.alpha_=0.;self.validation_=record_rates(zv)
        best=self.validation_['bacc'];self.gate_audit_=[]
        for alpha in self.alpha_grid:
            scores=zv+alpha*(av@self.coef_);metric=record_rates(scores)
            feasible=(metric['recall']+1e-12>=self.validation_['recall']-self.recall_tolerance and
                      metric['fpr']-1e-12<=self.validation_['fpr']+self.fpr_tolerance)
            self.gate_audit_.append(dict(alpha=alpha,feasible=bool(feasible),**metric))
            # Ties retain the smaller gate, including zero.
            if feasible and metric['bacc']>best+1e-12:self.alpha_=alpha;best=metric['bacc']
        self.optimization_={'success':True,'iterations':int(result.nit),'objective':float(result.fun)}
        return self

    def predict_log_odds(self,x,backbone_log_odds):
        if not hasattr(self,'coef_'):raise ValueError('Fit or restore the module before prediction')
        x=np.asarray(x,float);z=np.asarray(backbone_log_odds,float)
        if x.ndim!=2 or x.shape[1]!=len(self.stability_) or z.shape!=(len(x),):
            raise ValueError('Input dimensions mismatch')
        if not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Finite inputs required')
        if self.alpha_==0:return z.copy()
        h=self._transform(x)*self.stability_
        out=z+self.alpha_*(h@self.coef_[:-1]+self.coef_[-1])
        if not np.isfinite(out).all():raise ValueError('Residual prediction produced nonfinite values')
        return out

    def state(self):
        if not hasattr(self,'coef_'):raise ValueError('Fit or restore the module before saving')
        return {'name':'SGRF candidate v1','state_version':2,'feature_count':len(self.stability_),
                'feature_contract':{'count':len(self.stability_),'order':'same_column_order_as_fit'},
                'paired_training_groups':self.paired_groups_,'stability':self.stability_.tolist(),
                'transform':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self.transformer_.state().items()},
                'validation_record_count':self.validation_record_count_,
                'coef':self.coef_.tolist(),'alpha':self.alpha_,'l2':self.l2,
                'effect_cap':self.effect_cap,'min_paired_groups':self.min_paired_groups,
                'alpha_grid':list(self.alpha_grid),'recall_tolerance':self.recall_tolerance,
                'fpr_tolerance':self.fpr_tolerance,'validation_baseline':self.validation_,
                'gate_candidates':self.gate_audit_,'optimization':self.optimization_,
                'performance_status':'Structural tests only; no three-backbone performance claim.'}

    @classmethod
    def from_state(cls,state):
        """Restore a JSON-compatible version-2 inference checkpoint.

        Column order is an explicit caller contract; dimensional validation
        cannot detect a permutation of equally sized feature vectors.
        """
        if state.get('state_version')!=2:raise ValueError('Unsupported SGRF state version')
        d=state.get('feature_count')
        if isinstance(d,bool) or not isinstance(d,int) or d<1:
            raise ValueError('State requires a positive integer feature count')
        if state.get('feature_contract')!={'count':d,'order':'same_column_order_as_fit'}:
            raise ValueError('State feature contract mismatch')
        try:
            obj=cls(**{k:state[k] for k in ('l2','effect_cap','min_paired_groups','alpha_grid',
                                          'recall_tolerance','fpr_tolerance')})
            def vector(value,length,label):
                a=np.asarray(value,float)
                if a.shape!=(length,) or not np.isfinite(a).all():
                    raise ValueError('Invalid state '+label)
                return a.copy()
            obj.stability_=vector(state['stability'],d,'stability')
            obj.coef_=vector(state['coef'],d+1,'coefficients')
            if np.any((obj.stability_<0)|(obj.stability_>1)):
                raise ValueError('State stability must lie in [0,1]')
            obj.alpha_=float(state['alpha'])
            if obj.alpha_ not in obj.alpha_grid:raise ValueError('State alpha not in candidate grid')
            transform=state['transform']
            if transform['mode']!='enhanced':raise ValueError('Unsupported feature transform mode')
            obj.transformer_=FeatureTransform();obj.transformer_.mode='enhanced'
            for name in ('med','mad','mean','std'):
                setattr(obj.transformer_,name,vector(transform[name],d,'transform '+name))
            if np.any(obj.transformer_.mad<=0) or np.any(obj.transformer_.std<=0):
                raise ValueError('Transform scales must be positive')
            for key,attribute,minimum in [('paired_training_groups','paired_groups_',obj.min_paired_groups),
                                           ('validation_record_count','validation_record_count_',2)]:
                value=state[key]
                if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
                    raise ValueError('Invalid state '+key)
                setattr(obj,attribute,value)
            # Audit metadata does not enter inference, but is retained for an
            # exact checkpoint round trip without aliasing the caller's dict.
            from copy import deepcopy
            obj.validation_=deepcopy(state['validation_baseline'])
            obj.gate_audit_=deepcopy(state['gate_candidates'])
            obj.optimization_=deepcopy(state['optimization'])
        except (KeyError,TypeError,OverflowError) as exc:
            raise ValueError('Malformed SGRF state') from exc
        return obj
