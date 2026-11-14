"""Package the frozen annotations for transfer alongside raw recordings, outside Git."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-manifest', required=True, type=Path)
    parser.add_argument('--public-fixed-cycles', required=True, type=Path)
    parser.add_argument('--public-keep', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.public_fixed_cycles, float_precision='round_trip')
    keep = np.load(args.public_keep, allow_pickle=False)
    if (len(table)!=1364 or len(keep)!=1357 or len(np.unique(keep))!=1357
            or table.iloc[keep].label.value_counts().to_dict()!={0:794,1:563}):
        raise ValueError('Expected the frozen 1,364 candidates and 1,357 retained windows.')
    own = pd.read_csv(args.self_manifest)
    if len(own)!=462 or own.id.nunique()!=462 or own.physical_condition_key.nunique()!=156:
        raise ValueError('Expected the reviewed 462-recording self manifest.')
    table['retain'] = 0
    table.loc[keep,'retain'] = 1
    args.output.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output/'public_annotations.csv',index=False)
    shutil.copyfile(args.self_manifest,args.output/'self462_manifest.csv')
    print(args.output)


if __name__ == '__main__':
    main()
