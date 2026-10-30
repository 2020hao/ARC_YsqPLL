"""Independently recompute matched seed0 route metrics from saved predictions."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import _bootstrap as boot
from arc_model.evaluation.cycle_metrics import cycle_metrics

ROOT = boot.ROOT
OUT = ROOT / 'outputs/locked_routes_20261030'
DATASETS = ('continuous5m', 'zenodo5501527')
MODELS = ('ArcFENet', 'HYDRA', 'ArcFusionNet')
ROUTES = {'dual': ('baseline', 'calibration_only', 'dual_fixed20', 'dual_rcr'),
          'plugin': ('baseline', 'single_direct', 'single_retrained')}
PAIRS = (('dual_rcr', 'baseline'), ('dual_rcr', 'calibration_only'),
         ('dual_rcr', 'dual_fixed20'), ('single_direct', 'baseline'),
         ('single_retrained', 'baseline'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paired_interval(v):
    """Condition cluster bootstrap and descriptive paired sign-randomization test."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(20261030)
    draws = v[rng.integers(len(v), size=(10000, len(v)))].mean(1)
    low, high = np.quantile(draws, [.025, .975])
    if len(v) <= 13:
        signs = np.asarray(list(itertools.product([-1, 1], repeat=len(v))))
        p = float((abs((signs * v).mean(1)) >= abs(v.mean()) - 1e-15).mean())
    else:
        signs = rng.integers(0, 2, size=(100000, len(v))) * 2 - 1
        p = float((1 + (abs((signs * v).mean(1)) >= abs(v.mean()) - 1e-15).sum()) / 100001)
    return dict(delta_pp=float(v.mean() * 100), lo_pp=float(low * 100),
                hi_pp=float(high * 100), p=p, groups=len(v),
                positive_groups=int((v > 0).sum()), negative_groups=int((v < 0).sum()))


def summarize(final=False):
    config = yaml.safe_load((ROOT / 'configs/datasets.yaml').read_text())
    records, conditions, strata, receipts, missing, baseline_views = [], [], [], [], [], {}
    for ds in DATASETS:
        cfg = config[ds]
        table = pd.read_csv(ROOT / cfg['prepared_roots']['current_self_sync'] / 'cycles.csv')
        keep = np.load(ROOT / cfg['reviewed_indices'])
        splits = json.loads((ROOT / cfg['split_file']).read_text())
        for model in MODELS:
            for route, arms in ROUTES.items():
                for arm in arms:
                    files = [OUT / route / 'runs' / ds / model / f'f{f}' / f'{arm}.npz' for f in range(5)]
                    if not all(p.exists() for p in files):
                        missing.append(dict(dataset=ds, model=model, route=route, arm=arm,
                                            folds=sum(p.exists() for p in files)))
                        continue
                    parts = []
                    for fold, path in enumerate(files):
                        z = np.load(path)
                        ix = z['indices'].astype(int)
                        y = z['y']
                        margin = z['margin']
                        probability = z['probability']
                        expected = np.intersect1d(splits[fold]['test'], keep)
                        assert np.array_equal(ix, expected), path
                        assert np.array_equal(y, table.label.to_numpy()[ix]), path
                        assert np.isfinite(margin).all() and np.isfinite(probability).all(), path
                        assert np.all((probability >= 0) & (probability <= 1)), path
                        assert np.array_equal(margin >= 0, probability >= .5), path
                        parts.append(pd.DataFrame(dict(index=ix, y=y, margin=margin,
                                                       probability=probability, fold=fold)))
                        receipts.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(ix)))
                    allp = pd.concat(parts).sort_values('index')
                    assert np.array_equal(allp['index'], keep), (ds, model, arm)
                    ix = allp['index'].to_numpy()
                    y = allp.y.to_numpy()
                    score = allp.margin.to_numpy()
                    g = table.iloc[ix].condition_key.to_numpy()
                    ident = (ds, model)
                    if arm == 'baseline':
                        if ident in baseline_views:
                            previous = baseline_views[ident]
                            assert np.array_equal(previous >= 0, score >= 0)
                            assert np.max(abs(previous - score)) < 2e-4
                            continue
                        baseline_views[ident] = score
                    # Raw margins retain score resolution; no sigmoid clipping for ranking.
                    metric = cycle_metrics(y, score, g, threshold=0)
                    records.append(dict(dataset=ds, model=model, arm=arm, seed=0, **metric))
                    pred = score >= 0
                    for group in np.unique(g):
                        v = g == group
                        bacc = .5 * (pred[v & (y == 1)].mean() + (~pred[v & (y == 0)]).mean())
                        conditions.append(dict(dataset=ds, model=model, arm=arm, condition=group, BAcc=bacc))
                    valid = table.iloc[ix].kind.eq('pll').to_numpy()
                    for key, subset in [('valid_pll', valid), ('fixed20_fallback', ~valid)]:
                        yy, pp = y[subset], pred[subset]
                        tp, fp = int((pp & (yy == 1)).sum()), int((pp & (yy == 0)).sum())
                        tn, fn = int((~pp & (yy == 0)).sum()), int((~pp & (yy == 1)).sum())
                        recall = tp / (tp + fn) if tp + fn else np.nan
                        specificity = tn / (tn + fp) if tn + fp else np.nan
                        strata.append(dict(dataset=ds, model=model, arm=arm, stratum=key,
                                           N=len(yy), TP=tp, FP=fp, TN=tn, FN=fn,
                                           Recall=recall, FPR=1-specificity,
                                           BAcc_pooled=.5*(recall+specificity)))
    metrics = pd.DataFrame(records)
    group_frame = pd.DataFrame(conditions)
    pairs = []
    if len(group_frame):
        for (ds, model), sub in group_frame.groupby(['dataset', 'model']):
            t = sub.pivot(index='condition', columns='arm', values='BAcc')
            for candidate, reference in PAIRS:
                if candidate in t and reference in t:
                    assert t[[candidate, reference]].notna().all().all()
                    pairs.append(dict(dataset=ds, model=model, candidate=candidate, reference=reference,
                                      **paired_interval((t[candidate]-t[reference]).to_numpy())))
    paired = pd.DataFrame(pairs)
    if len(paired):
        paired['p_holm'] = np.nan
        for pair, t in paired.groupby(['candidate', 'reference']):
            # A partial family is not silently treated as a complete multiplicity family.
            if len(t) == 6:
                order = t.sort_values('p').index
                paired.loc[order, 'p_holm'] = np.maximum.accumulate(
                    [min(1., paired.loc[k, 'p'] * (6-i)) for i, k in enumerate(order)])
    if final:
        assert not missing, missing
        assert len(metrics) == 36 and len(paired) == 30
    prefix = '' if final else 'stage_'
    for name, frame in [('main_results', metrics), ('condition_metrics', group_frame),
                        ('paired_deltas', paired), ('reference_strata', pd.DataFrame(strata))]:
        frame.to_csv(OUT / f'{prefix}{name}.csv', index=False)
    result = dict(status='complete' if final else 'partial', stage='seed0_exploratory_fivefold',
                  metric_rows=len(metrics), expected_metric_rows=36, paired_rows=len(paired),
                  missing=missing, prediction_files=receipts, gpu0_used=False,
                  score_ranking='raw neural-logit difference or SVM margin; fixed zero threshold',
                  source='saved actual predictions, no historical C0 score inheritance')
    (OUT / f'{prefix}SUMMARY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if len(metrics):
        print(metrics.pivot(index=['dataset', 'model'], columns='arm', values='BAcc_condition').round(5).to_string())
    print('SUMMARY', result['status'], len(metrics), '/', 36, 'rows')
    return metrics, paired


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    summarize(parser.parse_args().final)
