import numpy as np
from arc_model.evaluation.cycle_metrics import cycle_metrics


def test_condition_equal_differs_from_pooled_and_threshold_is_explicit():
    # Tiny condition is perfect; large condition misses every arc. Weighting
    # the conditions equally gives .75; weighting every cycle gives .60 BAcc.
    y=np.r_[0,1,np.zeros(4),np.ones(4)].astype(int)
    score=np.r_[.1,.8,np.full(4,.1),np.full(4,.2)]
    groups=np.array(['small','small']+['large']*8)
    m=cycle_metrics(y,score,groups)
    assert m['BAcc_condition']==.75 and m['BAcc_pooled']==.6
    assert m['TP']==1 and m['FN']==4 and m['TN']==5 and m['FP']==0
    assert m['Recall']==.2 and m['Specificity']==1 and m['Accuracy']==.6
    assert m['F1']==1/3
    margin=score-.5
    assert cycle_metrics(y,margin,groups,threshold=0)==m
