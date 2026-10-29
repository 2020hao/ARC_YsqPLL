"""Read-only prepared arrays and record-group splits; no model fitting here."""
from pathlib import Path
import json
import numpy as np
from matplotlib import colormaps

class PreparedDataset:
    def __init__(self, prepared: str | Path, manifests: str | Path):
        self.root = Path(prepared)
        self.records = json.loads((Path(manifests) / 'records.json').read_text())
        self.splits = json.loads((Path(manifests) / 'splits.json').read_text())
        self.ledger = {r['record_id']: r for r in self.records}
        self._arrays = {}
        self.record_ids = self.array('record_ids')
        self.labels = np.array([self.ledger[int(r)]['label'] for r in self.record_ids])
        self._validate_scope()

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(self.root / (name + '.npy'), mmap_mode='r')
        return self._arrays[name]

    def _validate_scope(self):
        if not self.records or len(self.ledger) != len(self.records):
            raise ValueError('Manifest must contain unique record IDs')
        datasets = {r.get('dataset_id', 'new5m') for r in self.records}
        if len(datasets) != 1 or not datasets <= {'new5m', 'zenodo5501527'}:
            raise ValueError('Exactly one registered dataset per training run')
        public = datasets == {'zenodo5501527'}
        for r in self.records:
            if public:
                root = Path(__file__).resolve().parents[3] / 'data/public/zenodo5501527'
                source = Path(r.get('source_file', ''))
                # Resolve parent, preserving verified symlinked original files.
                if (source.parent.resolve() != root.resolve() or not source.name.startswith('C3_')
                        or r.get('sample_rate_hz') != 1000000
                        or r.get('label_status') != 'reviewed'
                        or r.get('label') not in (0, 1)):
                    raise ValueError('Public data require reviewed labels and registered 1 MHz C3 current')
                continue
            if (r.get('cohort') != 'main' or r.get('channel_zero_based') != 0
                    or r.get('distance_m') != 5 or r.get('sample_rate_hz') != 2000000
                    or not Path(r.get('source_file', '')).resolve().is_relative_to('/data/arc_data')
                    or 'NET9770_114' not in Path(r.get('source_file', '')).parts):
                raise ValueError('Current study accepts only /data/arc_data NET9770_114 CH0 at 5 m')
        ids = set(self.ledger)
        if set(self.record_ids) != ids:
            raise ValueError('Prepared window record IDs do not match the manifest')
        held_out = []
        for s in self.splits:
            train, test = set(s['train_records']), set(s['test_records'])
            if s['protocol'] != ('appliance_grouped_cv5' if public else 'condition_grouped_cv5') or train & test or train | test != ids:
                raise ValueError('Invalid real-appliance outer split')
            for field in (('condition_key', 'source_group', 'physical_run_id', 'appliance') if public else ('condition_key', 'source_group', 'physical_run_id')):
                if {self.ledger[i][field] for i in train} & {self.ledger[i][field] for i in test}:
                    raise ValueError(f'{field} leakage across the outer split')
            held_out.extend(test)
        if len(held_out) != len(ids) or set(held_out) != ids:
            raise ValueError('Each main record must be held out exactly once')

    def indices(self, fold):
        if not 0 <= fold < len(self.splits):
            raise ValueError(f'Fold must be between 0 and {len(self.splits)-1}')
        s = self.splits[fold]
        return tuple(np.flatnonzero(np.isin(self.record_ids, s[k]))
                     for k in ['train_records', 'test_records'])

    def physical(self, variant, mode, indices=None):
        a = self.array('features_low' if mode == 'enhanced_low' else 'features')
        return np.asarray(a[:, variant] if indices is None else a[indices, variant])

    def neural_inputs(self, name, variant, indices=None):
        index = slice(None) if indices is None else indices
        if name == 'ArcFENet':
            a = np.array(self.array('mbwvd')[index, variant], copy=True)
            lo, hi = a.min((1, 2), keepdims=True), a.max((1, 2), keepdims=True)
            a = (a-lo)/np.maximum(hi-lo, 1e-20)
            return colormaps['jet'](a)[..., :3].transpose(0, 3, 1, 2).astype('float32'), None
        if name != 'ArcFusionNet':
            raise ValueError(name)
        return (np.asarray(self.array('fusion_temporal')[index, variant]),
                np.asarray(self.array('fusion_spectrum')[index, variant]))
