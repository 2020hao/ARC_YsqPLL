"""Cycle metrics and conservative, coverage-audited event/normal interval metrics.

Window rows: stream_id, condition_key, start_seconds, end_seconds, label, score.
Intervals: stream_id,start_seconds,end_seconds. Event intervals are true arc
intervals, not commands. Labels and interval boundaries must be independently
reviewed. Times refer to original signal; latency here excludes compute time.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score


def cycle_metrics(frame, threshold):
    y=frame.label.to_numpy(int);s=frame.score.to_numpy(float)
    if not len(y) or not np.isfinite(s).all() or not set(y)<= {0,1}:
        raise ValueError('Need nonempty finite predictions and binary labels')
    pred=s>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    groups=[]
    for _,g in frame.groupby('condition_key'):
        rates=[float((g.loc[g.label==v,'score'].ge(threshold)==bool(v)).mean()) for v in (0,1) if (g.label==v).any()]
        if len(rates)==2:groups.append(np.mean(rates))
    return dict(windows=len(y),precision=float(p),recall=float(r),f1=float(f),
        false_positive_rate=float(fp/(tn+fp)) if tn+fp else None,
        average_precision=float(average_precision_score(y,s)) if len(set(y))==2 else None,
        pr_auc_definition='average_precision_step_integral',
        condition_equal_balanced_accuracy=float(np.mean(groups)) if groups else None,
        conditions_with_both_classes=len(groups),
        confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])


def interval_alarms(frame, start, end, threshold, consecutive, tolerance=2e-5):
    # Only fully contained windows; never bridge missing windows or overlap.
    g=frame[(frame.start_seconds>=start-tolerance)&(frame.end_seconds<=end+tolerance)].sort_values('start_seconds')
    run=0;latched=False;previous=None;alarms=[]
    for row in g.itertuples():
        if row.end_seconds<=row.start_seconds:raise ValueError('Invalid window duration')
        if previous is not None:
            gap=row.start_seconds-previous
            if gap < -tolerance:raise ValueError('Alarm evaluation requires nonoverlapping windows')
            if gap>tolerance:run=0;latched=False
        if row.score>=threshold:
            run+=1
            if run>=consecutive and not latched:alarms.append(float(row.end_seconds));latched=True
        else:run=0;latched=False
        previous=row.end_seconds
    duration=float((g.end_seconds-g.start_seconds).sum())
    return g,alarms,duration


def evaluate(frame, threshold, events=None, normal_intervals=None, consecutive=9, deadline=.2):
    required={'stream_id','condition_key','start_seconds','end_seconds','label','score'}
    if not required<=set(frame):raise ValueError(f'Missing columns: {required-set(frame)}')
    if frame.duplicated(['stream_id','start_seconds','end_seconds']).any():raise ValueError('Duplicate windows: evaluate seeds/folds as separate runs')
    if consecutive < 1 or deadline <= 0:raise ValueError('Positive alarm settings required')
    if not np.isfinite(frame[['start_seconds','end_seconds','score']].to_numpy()).all():raise ValueError('Nonfinite prediction data')
    for table in (events,normal_intervals):
        if table is None:continue
        if not {'stream_id','start_seconds','end_seconds'}<=set(table):raise ValueError('Missing interval fields')
        if not np.isfinite(table[['start_seconds','end_seconds']].to_numpy()).all() or (table.end_seconds<=table.start_seconds).any():raise ValueError('Invalid intervals')
        for _,group in table.groupby('stream_id'):
            group=group.sort_values('start_seconds')
            if np.any(group.start_seconds.to_numpy()[1:] < group.end_seconds.to_numpy()[:-1]):raise ValueError('Overlapping intervals double-count exposure')
    result={'cycle':cycle_metrics(frame,threshold), 'alarm_policy':dict(consecutive_cycles=consecutive,deadline_seconds=deadline,processing_latency_included=False)}
    event_rows=[]
    if events is not None:
        for event in events.itertuples():
            stop=min(event.end_seconds,event.start_seconds+deadline)
            g,alarms,duration=interval_alarms(frame[frame.stream_id==event.stream_id],event.start_seconds,stop,threshold,consecutive)
            if len(g) and not (g.label==1).all():raise ValueError('Arc event interval contains normal labels')
            # Allow boundary rounding of < one full cycle at either edge; no internal gaps.
            gaps=np.diff(g.start_seconds.to_numpy())-(g.end_seconds-g.start_seconds).to_numpy()[:-1]
            covered=len(g)>0 and np.all(np.abs(gaps)<=2e-5) and g.start_seconds.iloc[0]-event.start_seconds<.025 and stop-g.end_seconds.iloc[-1]<.025
            eligible=covered and event.end_seconds-event.start_seconds>=deadline
            event_rows.append(dict(stream_id=event.stream_id,start_seconds=float(event.start_seconds),eligible=bool(eligible),detected=bool(alarms) if eligible else None,latency_seconds=alarms[0]-event.start_seconds if eligible and alarms else None))
        eligible=[e for e in event_rows if e['eligible']];lat=[e['latency_seconds'] for e in eligible if e['detected']]
        result['event']=dict(available=bool(eligible),events_total=len(event_rows),events_evaluable=len(eligible),events_excluded=len(event_rows)-len(eligible),
            detection_rate_by_deadline=len(lat)/len(eligible) if eligible else None,miss_rate_by_deadline=1-len(lat)/len(eligible) if eligible else None,
            latency_median_seconds=float(np.median(lat)) if lat else None,latency_p95_seconds=float(np.percentile(lat,95)) if lat else None,
            latency_population='detected eligible events only; misses reported separately',details=event_rows)
    else:result['event']=dict(available=False,reason='No independently reviewed onset/end intervals')
    if normal_intervals is not None:
        count=0;seconds=0.;trials=[];durations=[]
        for interval in normal_intervals.itertuples():
            g,alarms,d=interval_alarms(frame[frame.stream_id==interval.stream_id],interval.start_seconds,interval.end_seconds,threshold,consecutive)
            if len(g) and not (g.label==0).all():raise ValueError('Normal evaluation interval contains arc labels')
            gaps=np.diff(g.start_seconds.to_numpy())-(g.end_seconds-g.start_seconds).to_numpy()[:-1]
            complete=len(g)>0 and np.all(np.abs(gaps)<=2e-5) and g.start_seconds.iloc[0]-interval.start_seconds<.025 and interval.end_seconds-g.end_seconds.iloc[-1]<.025
            if complete:count+=len(alarms);seconds+=d;trials.append(bool(alarms));durations.append(d)
        result['normal']=dict(available=bool(trials),false_alarm_episodes=count,observed_normal_seconds=seconds,
            false_alarm_episodes_per_hour=count*3600/seconds if seconds else None,
            trial_false_alarm_proportion=float(np.mean(trials)) if trials else None,
            trial_duration_seconds=durations,
            trials_evaluable=len(trials),trials_excluded=len(normal_intervals)-len(trials),
            interpretation='Observed rate only; report short exposure explicitly')
    else:result['normal']=dict(available=False,reason='No reviewed continuous normal trial intervals')
    return result
