"""Portable entry for candidate design, raw preparation, feature extraction and smoke checks."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from arc_model.data.current_recording import read_current
from arc_model.methods.phase_representation import phase_features
from arc_model.methods.cycle_dynamics import deterministic_phase_shifts
from arc_model.preprocessing.phase_windows import prepare_phase_windows


def _new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Nonempty output refused: {path}; use a new version directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_features(energy, metadata, out, mode):
    shifts = (
        deterministic_phase_shifts("phase_module_v1", len(energy), energy.shape[-1])
        if mode == "shifted"
        else None
    )
    valid = (
        metadata.phase_valid.to_numpy()
        if "phase_valid" in metadata
        else metadata.kind.eq("pll").to_numpy()
    )
    batch = phase_features(energy, phase_valid=valid, mode=mode, shifts=shifts)
    np.savez(
        out / "phase_features.npz",
        values=batch.values,
        names=batch.names,
        valid=batch.valid,
        phase_valid=batch.phase_valid,
        mode=mode,
    )
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("design", help="Print candidate specification; does not train")
    p = sub.add_parser(
        "prepare-record", help="Current-only raw preparation; no class labels generated"
    )
    p.add_argument("--current", required=True)
    p.add_argument("--sample-rate", type=int, required=True)
    p.add_argument("--reference", choices=["fixed20", "sogi", "rcr", "cft"], default="rcr")
    p.add_argument("--output", required=True)
    p = sub.add_parser("features", help="Read-only existing native-HF energy and window metadata")
    p.add_argument("--energy", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--mode", choices=["phase", "phase_free", "shifted"], default="phase")
    p.add_argument("--output", required=True)
    sub.add_parser(
        "model-smoke",
        help="CPU baseline parity, fusion gradient and shape checks; not detector evaluation",
    )
    args = parser.parse_args(argv)
    if args.command == "design":
        print(
            json.dumps(
                dict(
                    name="current_synchronous_phase_representation",
                    status="candidate_not_detection_validated",
                    feature_dimension=10,
                    bands_hz=[[10000, 100000], [100000, 400000]],
                    controls=["phase_free", "fixed20", "independent_cycle_shift"],
                    fusion="pre_classifier_concatenation",
                    auto_train=False,
                ),
                indent=2,
            )
        )
    elif args.command == "prepare-record":
        record = read_current(args.current, args.sample_rate)
        prepared = prepare_phase_windows(
            record.current, record.sample_rate_hz, reference=args.reference
        )
        out = _new_output(args.output)
        for name in ["wave", "native", "energy"]:
            np.save(out / (name + ".npy"), getattr(prepared, name))
        metadata = prepared.metadata.copy()
        for key in ["start_seconds", "end_seconds", "available_seconds", "anchor_seconds"]:
            metadata[key] += record.time_origin_s
        metadata.to_csv(out / "windows.csv", index=False)
        prepared.clock.to_csv(out / "reference_clock.csv", index=False)
        batch = _save_features(prepared.energy, metadata, out, "phase")
        digest = hashlib.sha256()
        with record.source.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        receipt = dict(
            status="prepared_unlabeled",
            source=str(record.source),
            source_sha256=digest.hexdigest(),
            sample_rate_hz=record.sample_rate_hz,
            reference=args.reference,
            time_origin_s=record.time_origin_s,
            windows=len(batch.values),
            feature_dimension=10,
            source_channel="NET9770_114_CH0"
            if record.source.suffix.lower() == ".bin"
            else "current",
            labels_generated=False,
            backbone_representations_materialized=False,
        )
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
    elif args.command == "features":
        energy = np.load(args.energy, mmap_mode="r", allow_pickle=False)
        metadata = pd.read_csv(args.metadata)
        if len(metadata) != len(energy):
            raise ValueError("Energy and metadata rows differ")
        batch = phase_features(
            energy,
            phase_valid=metadata.kind.eq("pll").to_numpy()
            if "kind" in metadata
            else metadata.phase_valid.to_numpy(),
            mode=args.mode,
            shifts=deterministic_phase_shifts("phase_module_v1", len(energy), energy.shape[-1])
            if args.mode == "shifted"
            else None,
        )
        out = _new_output(args.output)
        np.savez(
            out / "phase_features.npz",
            values=batch.values,
            valid=batch.valid,
            names=batch.names,
            phase_valid=batch.phase_valid,
            mode=batch.mode,
        )
        print(
            json.dumps(
                dict(
                    status="features_only_not_trained", windows=len(batch.values), names=batch.names
                ),
                indent=2,
            )
        )
    else:
        from arc_model.models.phase_adapter import PhaseFeatureAdapter
        from arc_model.models.factory import make_model
        import torch

        torch.set_num_threads(2)
        result = []
        for name in ["ArcFENet", "HYDRA", "ArcFusionNet"]:
            torch.manual_seed(41)
            base = torch.nn.Linear(448, 1) if name == "HYDRA" else make_model(name)
            model = PhaseFeatureAdapter(base, name).eval()
            x = (
                torch.randn(2, 448)
                if name == "HYDRA"
                else torch.randn(2, 3, 100, 100)
                if name == "ArcFENet"
                else torch.randn(2, 1, 3, 250)
            )
            spectral = torch.randn(2, 1, 32) if name == "ArcFusionNet" else None
            f = torch.randn(2, 10)
            with torch.no_grad():
                original = base(x) if name == "HYDRA" else base.eval()(x, spectral)
                disabled = model(x, spectral, enabled=False)
                torch.testing.assert_close(original, disabled, rtol=0, atol=0)
            model.zero_grad()
            model(x, spectral, f).sum().backward()
            if model.fusion_head.weight.grad is None:
                raise AssertionError("No fusion gradient")
            result.append(
                dict(
                    model=name,
                    disabled_baseline_exact=True,
                    shape=list(disabled.shape),
                    gradient=True,
                )
            )
        print(json.dumps(dict(status="CPU_interface_smoke_only", models=result), indent=2))


if __name__ == "__main__":
    main()
