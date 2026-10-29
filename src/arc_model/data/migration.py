"""Import existing real-appliance artifacts, preserving original IDs and hashes."""
from pathlib import Path
import json
import shutil
import numpy as np
from arc_model.utils.io import save_json, sha256

ARRAY_SOURCES = {
    'record_ids': 'record_ids.npy', 'features': 'features.npy',
    'features_low': 'features_low.npy', 'fusion_temporal': 'fusion_temporal.npy',
    'fusion_spectrum': 'fusion_spectrum.npy', 'wave_50000': 'inputs/wave_50000.npy',
    'wave_250000': 'inputs/wave_250000.npy', 'mbwvd': 'inputs/mbwvd.npy',
    **{f'hydra_features_seed{s}': f'hydra_features_seed{s}.npy' for s in range(3)},
}


def import_inputs(legacy, prepared, manifests):
    legacy, prepared, manifests = map(Path, (legacy, prepared, manifests))
    if prepared.exists() and any(prepared.iterdir()):
        raise FileExistsError(f'Prepared directory is not empty: {prepared}')
    records = [r for r in json.loads((legacy/'record_manifest.json').read_text()) if r['cohort']=='main']
    splits = [s for s in json.loads((legacy/'splits.json').read_text()) if s['protocol']=='leave_appliance_out']
    old_ids = np.load(legacy/'record_ids.npy')
    keep = np.flatnonzero(np.isin(old_ids, [r['record_id'] for r in records]))
    prepared.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    np.save(prepared/'original_window_indices.npy', keep)
    audit = dict(source=str(legacy), scope='real_appliances_only', records=len(records),
                 windows=len(keep), folds=len(splits), arrays={})
    for name, relative in ARRAY_SOURCES.items():
        source = legacy/relative
        original = np.load(source, mmap_mode='r')
        if len(original) != len(old_ids):
            raise ValueError(f'Window alignment mismatch: {source}')
        target = prepared/(name+'.npy')
        filtered = np.lib.format.open_memmap(target, mode='w+', dtype=original.dtype,
                                            shape=(len(keep), *original.shape[1:]))
        for offset in range(0,len(keep),128):
            filtered[offset:offset+128] = original[keep[offset:offset+128]]
        filtered.flush()
        audit['arrays'][name] = dict(shape=list(filtered.shape), dtype=str(filtered.dtype),
                                    source_sha256=sha256(source), sha256=sha256(target))
        del filtered
    save_json(manifests/'records.json', records)
    save_json(manifests/'splits.json', splits)
    save_json(manifests/'import_provenance.json', audit)
    return audit


def import_checkpoints(legacy, prepared, destination, selection='representative'):
    legacy, prepared, destination = map(Path, (legacy, prepared, destination))
    original = np.load(prepared/'original_window_indices.npy')
    mapping = {int(old): new for new,old in enumerate(original)}
    selected = []
    if selection == 'representative':
        for model in ('HYDRA','ArcFENet','ArcFusionNet'):
            for mode in ('base','enhanced'):
                selected.append(legacy/'runs'/f'{model}_v1_{mode}_s0'/'fold0.json')
    elif selection == 'all':
        selected = sorted((legacy/'runs').glob('*/fold*.json'))
    else:
        raise ValueError(selection)
    audit=[]
    for metadata in selected:
        meta=json.loads(metadata.read_text())
        if meta.get('protocol') != 'leave_appliance_out':
            continue
        dest=destination/metadata.parent.name
        dest.mkdir(parents=True,exist_ok=True)
        stem=metadata.stem
        if (dest/metadata.name).exists():
            raise FileExistsError(dest/metadata.name)
        copied={}
        suffixes=['.npz'] if meta['model']=='HYDRA' else ['.pt','_preprocessing.npz']
        for suffix in suffixes:
            source=metadata.parent/(stem+suffix)
            shutil.copy2(source,dest/source.name)
            copied[source.name]=sha256(source)
        source=metadata.parent/(stem+'_windows.npz')
        with np.load(source) as values:
            arrays={k:values[k] for k in values.files}
        arrays['original_indices']=arrays['indices'].copy()
        arrays['indices']=np.array([mapping[int(i)] for i in arrays['indices']])
        np.savez_compressed(dest/source.name,**arrays)
        meta['migration']=dict(source=str(metadata), source_sha256=sha256(metadata),
                               window_indices='remapped; original_indices retained', weights='unchanged')
        save_json(dest/metadata.name,meta)
        audit.append(dict(metadata=str(dest/metadata.name), source=str(metadata), unchanged_artifacts=copied))
    save_json(destination/'import_manifest.json',audit)
    return audit
