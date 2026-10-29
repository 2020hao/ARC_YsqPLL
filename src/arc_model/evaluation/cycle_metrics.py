"""Binary cycle metrics: arc is positive; condition and pooled weights explicit."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, roc_auc_score,
)


def cycle_metrics(labels, scores, conditions, threshold=.5):
    y=np.asarray(labels);s=np.asarray(scores);g=np.asarray(conditions)
    if y.ndim!=1 or s.shape!=y.shape or g.shape!=y.shape or not len(y):
        raise ValueError('Nonempty corresponding 1D labels/scores/conditions required')
    if set(np.unique(y))!={0,1} or not np.isfinite(s).all():
        raise ValueError('Both binary classes and finite scores required')
    p=s>=threshold;tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    recall=tp/(tp+fn);specificity=tn/(tn+fp);group_bacc=[]
    for key in np.unique(g):
        yy=y[g==key];pp=p[g==key]
        if set(np.unique(yy))!={0,1}:
            raise ValueError('Each condition must contain both classes for condition BAcc')
        group_bacc.append(.5*(pp[yy==1].mean()+(~pp[yy==0]).mean()))
    return dict(N=len(y),groups=len(group_bacc),BAcc_condition=float(np.mean(group_bacc)),
        BAcc_pooled=.5*(recall+specificity),Accuracy=accuracy_score(y,p),
        Precision=precision_score(y,p,zero_division=0),Recall=recall,Specificity=specificity,
        FPR=1-specificity,FNR=1-recall,F1=f1_score(y,p,zero_division=0),
        Macro_F1=f1_score(y,p,average='macro',zero_division=0),
        ROC_AUC=roc_auc_score(y,s),AP=average_precision_score(y,s),MCC=matthews_corrcoef(y,p),
        TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn))
