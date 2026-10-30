"""PILT-style YAML catalog and routing only; no shared network components."""

from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from importlib import import_module
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS = ROOT / "configs/models.yaml"


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key, value in node.value:
        k = loader.construct_object(key, deep=deep)
        if k in result:
            raise ValueError("Duplicate YAML key: " + str(k))
        if k in ["<<", "extends", "_base_"]:
            raise ValueError("Independent model nodes cannot inherit/merge")
        result[k] = loader.construct_object(value, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    implementation: str
    role: str
    backbone: str
    values: dict


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def load_catalog(path=DEFAULT_MODELS):
    document = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(document, dict) or len(document) != 9:
        raise ValueError("Exactly nine independent model nodes required")
    expected = json.loads((Path(__file__).parent / "catalog.json").read_text())["models"]
    mapping = {r["config_key"]: r for r in expected}
    result = {}
    if set(document) != set(mapping):
        raise ValueError("Model catalog keys differ from the nine registered implementations")
    for name, node in document.items():
        if not isinstance(node, dict):
            raise ValueError("Each model node must be a complete mapping")
        for key in [
            "implementation",
            "role",
            "backbone",
            "model_kwargs",
            "model_training_param",
            "input_contract",
            "feature_module",
            "precision",
            "evaluation",
        ]:
            if key not in node:
                raise ValueError(name + " missing " + key)
        reference = mapping[name]
        if node["implementation"] != reference["key"] or any(
            node[k] != reference[k] for k in ["role", "backbone"]
        ):
            raise ValueError(name + " implementation identity mismatch")
        if (
            node["evaluation"]["datasets"] != ["continuous5m", "zenodo5501527"]
            or node["evaluation"]["grouped_outer_folds"] != 5
        ):
            raise ValueError("Current dual-dataset grouped5fold protocol required")
        kwargs = node["model_kwargs"]
        params = node["model_training_param"]
        feature = node["feature_module"]
        if kwargs.get("extra", 0) != 0:
            raise ValueError(
                "Native extra branch must stay disabled; dual branch has an explicit independent implementation"
            )
        if node["role"] == "baseline" and (
            feature["feature_dim"] != 0 or feature["mode"] != "none"
        ):
            raise ValueError("No research feature module in baseline")
        if node["role"] == "plugin" and (
            feature["feature_dim"] != 0 or feature["auxiliary_branch"]
        ):
            raise ValueError("Input frontend cannot silently add an auxiliary branch")
        if node["role"] == "dual" and feature["feature_dim"] != 36:
            raise ValueError("This candidate implements the explicit common36D definition")
        if node["backbone"] == "hydra":
            if params["optimizer"] != "HiGHS_L1_hinge" or params["lambda_l1"] <= 0:
                raise ValueError("Invalid paper-FD SVM recipe")
            if (
                kwargs["kernel_length"] < 2
                or kwargs["kernel_length"] >= 1000
                or min(kwargs["k"], kwargs["g"], kwargs["stride"]) < 1
            ):
                raise ValueError("Invalid HYDRA transform dimensions")
            width = (
                2
                * (int(math.log2(999 / (kwargs["kernel_length"] - 1))) + 1)
                * kwargs["g"]
                * kwargs["k"]
            )
            if node["input_contract"]["input_shape"] != ["N", width]:
                raise ValueError("HYDRA feature contract must match its configured transform")
        else:
            if (
                params["optimizer"] not in ["Adam", "AdamW"]
                or params["epochs"] <= 0
                or params["batch_size"] <= 0
                or params["lr"] <= 0
                or params["wd"] < 0
            ):
                raise ValueError("Invalid neural training recipe")
            if params["scheduler"] not in ["none", "cosine"]:
                raise ValueError("Unsupported scheduler")
        result[name] = ModelSpec(
            name, node["implementation"], node["role"], node["backbone"], deepcopy(node)
        )
    return result


def create_model(name, *, config_path=DEFAULT_MODELS, checkpoint=None):
    spec = load_catalog(config_path)[name]
    if spec.role == "baseline":
        local = import_module("arc_model.implementations." + spec.implementation + ".configuration")
        local.load_config(config_path)
    module = import_module("arc_model.implementations." + spec.implementation + ".model")
    cls = module.Model
    kwargs = deepcopy(spec.values["model_kwargs"])
    inspect.signature(cls).bind_partial(**kwargs)
    if spec.backbone == "hydra":
        return cls.load(checkpoint, **kwargs) if checkpoint is not None else cls(**kwargs)
    model = cls(**kwargs)
    if checkpoint is not None:
        import torch

        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["list", "validate", "show", "smoke"])
    p.add_argument("--model")
    p.add_argument("--config", default=str(DEFAULT_MODELS))
    a = p.parse_args()
    specs = load_catalog(a.config)
    if a.action == "show":
        if a.model not in specs:
            raise ValueError("Select one registered --model")
        print(yaml.safe_dump(specs[a.model].values, allow_unicode=True, sort_keys=False))
        return
    if a.action in ["list", "validate"]:
        print(
            json.dumps(
                [
                    dict(
                        name=s.name,
                        implementation=s.implementation,
                        role=s.role,
                        performance_status=s.values["performance_status"],
                    )
                    for s in specs.values()
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    import torch
    import numpy as np

    torch.set_num_threads(2)
    rows = []
    for name, s in specs.items():
        model = create_model(name, config_path=a.config)
        if s.backbone == "hydra":
            feature = model.extract(np.ones((2, 1000), dtype="float32"))
            width = feature.shape[1]
            model.state = dict(
                weight=np.zeros(width),
                intercept=np.asarray(0.0),
                mu=np.zeros(width, dtype="float32"),
                sigma=np.ones(width, dtype="float32"),
                epsilon=np.ones(width, dtype="float32"),
            )
            shape = list(model.decision_function(feature).shape)
        else:
            model.eval()
            with torch.no_grad():
                shape = (
                    list(model(torch.zeros(2, 3, 100, 100)).shape)
                    if s.backbone == "arc_fenet"
                    else list(model(torch.zeros(2, 1, 3, 250), torch.zeros(2, 1, 32)).shape)
                )
        rows.append(dict(name=name, shape=shape))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
