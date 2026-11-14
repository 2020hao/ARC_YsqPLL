"""Compare two complete correction metric files, with an explicit tolerance in pp."""
import argparse
from pathlib import Path

import pandas as pd

KEYS = ['dataset', 'model', 'arm']
METRICS = ['Acc', 'BAcc', 'Precision', 'Recall', 'FPR', 'F1']


def compare(reference, candidate, tolerance):
    frozen = pd.read_csv(reference).set_index(KEYS)
    rebuilt = pd.read_csv(candidate).set_index(KEYS)
    if not frozen.index.is_unique or not rebuilt.index.is_unique or rebuilt.empty:
        raise ValueError('Expected nonempty results with unique dataset/model/arm rows.')
    expected = frozen.loc[frozen.index.get_level_values('dataset').isin(
        rebuilt.index.get_level_values('dataset'))]
    if set(expected.index) != set(rebuilt.index):
        raise ValueError('Candidate result is incomplete or contains unexpected rows.')
    if not (expected['N'].sort_index().eq(rebuilt['N'].sort_index()).all()
            and expected['seed'].sort_index().eq(rebuilt['seed'].sort_index()).all()):
        raise ValueError('Sample counts or seeds differ.')
    delta = (rebuilt[METRICS]-expected[METRICS])*100
    if not delta.notna().all().all():
        raise ValueError('Missing metrics.')
    print(delta.round(5).to_string())
    print(f'Maximum absolute difference: {delta.abs().max().max():.6f} pp')
    return bool((delta.abs() <= tolerance).all().all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--tolerance-pp', type=float, default=0.0)
    args = parser.parse_args()
    if args.tolerance_pp < 0:
        parser.error('Tolerance must be nonnegative.')
    return 0 if compare(args.reference,args.candidate,args.tolerance_pp) else 1


if __name__=='__main__':
    raise SystemExit(main())
