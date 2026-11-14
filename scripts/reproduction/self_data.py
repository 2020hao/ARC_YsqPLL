"""Rebuild the frozen 462-recording input cohort from reviewed CH0 BINs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from self462_extract_20261105 import one as extract  # noqa: E402
import prepare_self462_1m_baselines_20261105 as one_mhz  # noqa: E402
import prepare_self462_12800_20261105 as low_rate  # noqa: E402
import run_self462_feature_study_20261105 as native  # noqa: E402
from prepare_self462_1m_signals_20261105 import one as extract_one_mhz  # noqa: E402


def resolve_recordings(manifest, recordings_root):
    """Resolve original paths or relocate the exported BIN tree by basename."""
    table = pd.read_csv(manifest, dtype={'id': str})
    if len(table) != 462 or table.id.nunique() != 462 or table.physical_condition_key.nunique() != 156:
        raise ValueError('Supply the reviewed 462-recording manifest (156 conditions).')
    if set(table.target_load) != {'吹风机', '热水壶', '电磁炉', '电视'}:
        raise ValueError('The manifest has different target loads.')
    files = {}
    if recordings_root is not None:
        for file in recordings_root.rglob('NET9770_114*.bin'):
            if file.name in files:
                raise ValueError(f'Ambiguous BIN basename: {file.name}')
            files[file.name] = file.resolve()
    for i, name in table.bin_114.items():
        path = Path(name)
        found = files.get(path.name) if recordings_root is not None else path
        if found is None or not found.is_file():
            raise FileNotFoundError(path)
        table.loc[i, 'bin_114'] = str(found.resolve())
    return table


def extract_record(job):
    row, native_dir, low_dir, one_mhz_dir = job
    receipt = extract((row, native_dir/"records"))
    signal = extract_one_mhz((row, one_mhz_dir/"records"))
    if receipt['CH0_sha256'] != signal['source_CH0_sha256']:
        raise ValueError(f"Different raw source for {row['id']}")
    # These are the original two-stage causal low-rate inputs.
    from prepare_self462_12800_signals_20261105 import one as extract_low
    extract_low((row, low_dir/"records"))
    return row['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--recordings-root', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    table = resolve_recordings(args.manifest, args.recordings_root)
    for folder in (native.OUT, low_rate.OUT, one_mhz.OUT):
        folder.mkdir(parents=True, exist_ok=True)
        new = table.to_csv(index=False)
        path = folder / 'manifest.csv'
        if path.exists() and path.read_text(encoding='utf-8-sig').replace('\r\n','\n') != new:
            raise FileExistsError(f'Existing work directory has a different manifest: {path}')
        path.write_text(new)
    jobs = (
        (row, native.OUT/'data', low_rate.OUT/'signals', one_mhz.OUT/'signals')
        for row in table.to_dict('records')
    )
    with ProcessPoolExecutor(args.workers) as pool:
        for index, record_id in enumerate(pool.map(extract_record, jobs), 1):
            print(f'{index}/462 {record_id}', flush=True)
    native.prepare()
    for stage, cfgname in ((low_rate, 'self462_12800'), (one_mhz, 'self462_1m_baselines')):
        config = yaml.safe_load((SCRIPTS.parent/'configs/modules'/f'{cfgname}_20261105.yaml').read_text())
        config['execution']['preparation_workers'] = args.workers
        stage.prepare(config)
    actual = pd.read_csv(one_mhz.OUT/'cycles.csv')
    assert len(actual) == 125051 and actual.label.value_counts().to_dict()=={0:61896,1:63155}
    print('Prepared self462 with 125,051 retained windows.')


if __name__ == '__main__':
    main()
