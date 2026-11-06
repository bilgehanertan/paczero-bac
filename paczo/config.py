import dataclasses
import json
import math
import os
import typing
from pathlib import Path
from typing import Literal

import yaml

from paczo.run_paczo import PACZeroArguments

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


class ConfigError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class DPAggZOEnv:
    MODEL: str
    TASK: Literal["SST2", "SQuAD"]
    MODE: Literal["ft", "lora"]
    LR: float
    EPS: float
    SEED: int
    TRAIN: int
    DEV: int
    EVAL: int
    STEPS: int
    EVAL_STEPS: int
    DP_EPS: float
    DP_CLIP: float
    DP_SAMPLE_RATE: float
    N: int


TRAINING_ARGUMENTS = {
    "learning_rate": float,
    "max_steps": int,
    "per_device_train_batch_size": int,
    "seed": int,
    "logging_steps": int,
    "eval_strategy": Literal["no", "steps"],
    "eval_steps": int,
    "save_strategy": Literal["no", "steps"],
    "save_steps": int,
    "save_total_limit": int,
    "load_best_model_at_end": bool,
    "metric_for_best_model": str,
    "greater_is_better": bool,
    "lr_scheduler_type": str,
    "lr_scheduler_kwargs": dict,
}
DERIVED = ("output_dir", "pac_diag_dir", "result_file")
KEYS = {
    "paczo": {k: t for k, t in PACZeroArguments.__annotations__.items() if k not in DERIVED} | TRAINING_ARGUMENTS,
    "dpaggzo": dict(DPAggZOEnv.__annotations__),
}
DEFAULTS = {f.name: f.default for f in dataclasses.fields(PACZeroArguments)}


@dataclasses.dataclass(frozen=True)
class Base:
    runner: Literal["paczo", "dpaggzo"]
    params: dict
    requires: tuple = ()
    setting: bool = False


@dataclasses.dataclass(frozen=True)
class Variant:
    base: str
    params: dict
    requires: tuple = ()


@dataclasses.dataclass(frozen=True)
class Run:
    id: str
    file: Path
    runner: str
    setting: str | None
    variant: str
    params: dict
    sources: dict


class _Loader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(k, deep=deep) for k, _ in node.value]
        dup = sorted({str(k) for k in keys if keys.count(k) > 1})
        if dup:
            raise ConfigError(f"{node.start_mark.name}:{node.start_mark.line + 1}: duplicate key {', '.join(dup)}")
        return super().construct_mapping(node, deep)


def _yaml(path):
    with open(path) as fh:
        return yaml.load(fh, Loader=_Loader)


def _mapping(value, where):
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {value!r}")
    return value


def _keys(value, allowed, required, where):
    _mapping(value, where)
    unknown = sorted(set(value) - set(allowed))
    missing = sorted(set(required) - set(value))
    if unknown:
        raise ConfigError(f"{where}: unknown key {', '.join(map(str, unknown))}")
    if missing:
        raise ConfigError(f"{where}: missing key {', '.join(missing)}")
    return value


def _check(value, hint, where):
    if value is None:
        raise ConfigError(f"{where}: null is not allowed, omit the key to keep the default")
    if typing.get_origin(hint) is Literal:
        if value not in typing.get_args(hint) or isinstance(value, bool):
            raise ConfigError(f"{where}: {value!r} is not one of {', '.join(map(str, typing.get_args(hint)))}")
        return value
    if hint is float and isinstance(value, str):
        try:
            float(value)
        except ValueError:
            pass
        else:
            raise ConfigError(f"{where}: YAML reads {value} as a string, write it with a dot and a signed exponent, as in 1.0e-4")
    ok = {
        bool: isinstance(value, bool),
        int: isinstance(value, int) and not isinstance(value, bool),
        float: isinstance(value, (int, float)) and not isinstance(value, bool),
        str: isinstance(value, str),
        dict: isinstance(value, dict),
    }[hint]
    if not ok:
        raise ConfigError(f"{where}: expected {hint.__name__}, got {value!r}")
    if hint is float:
        return float(value)
    if hint is dict:
        return {str(k): _check(v, float, f"{where}.{k}") for k, v in value.items()}
    return value


def _params(value, runner, where):
    keys = KEYS[runner]
    out = {}
    for k, v in _mapping(value, where).items():
        if k not in keys:
            raise ConfigError(f"{where}: {k} is not a {runner} parameter")
        out[k] = _check(v, keys[k], f"{where}: {k}")
    return out


def _names(value, where):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}: expected a list of names, got {value!r}")
    return tuple(value)


def load_registry():
    doc = _keys(_yaml(CONFIGS / "variants.yaml"), ("bases", "variants"), ("bases", "variants"), "variants.yaml")
    bases = {}
    for name, b in _mapping(doc["bases"], "variants.yaml: bases").items():
        where = f"variants.yaml: base {name}"
        _keys(b, ("runner", "setting", "requires", "params"), ("runner", "params"), where)
        runner = _check(b["runner"], Literal["paczo", "dpaggzo"], f"{where}: runner")
        bases[name] = Base(
            runner,
            _params(b["params"], runner, f"{where}: params"),
            _names(b.get("requires", []), f"{where}: requires"),
            _check(b.get("setting", False), bool, f"{where}: setting"),
        )
    variants = {}
    for name, v in _mapping(doc["variants"], "variants.yaml: variants").items():
        where = f"variants.yaml: variant {name}"
        _keys(v, ("base", "requires", "params"), ("base", "params"), where)
        if v["base"] not in bases:
            raise ConfigError(f"{where}: unknown base {v['base']!r}")
        variants[name] = Variant(
            v["base"],
            _params(v["params"], bases[v["base"]].runner, f"{where}: params"),
            _names(v.get("requires", []), f"{where}: requires"),
        )
    settings = {
        name: _params(p, "paczo", f"settings.yaml: {name}")
        for name, p in _mapping(_yaml(CONFIGS / "settings.yaml"), "settings.yaml").items()
    }
    return settings, variants, bases


def _cross_checks(params, where):
    def eff(k):
        return params.get(k, DEFAULTS[k])

    rules = [
        (eff("trainer") == "zo" and eff("max_steps") <= 0, "max_steps must be positive, one step is one full-batch update"),
        (eff("pac_m") < 2 or eff("pac_m") % 2, f"pac_m must be even and at least 2, got {eff('pac_m')}"),
        (eff("pac_k") < 1, f"pac_k must be at least 1, got {eff('pac_k')}"),
        (eff("pac_k") > 1 and not eff("pac_zo_third_pass"), "pac_k > 1 needs pac_zo_third_pass, the two-pass train_loss reuses one direction"),
        (not 0.0 < eff("pac_clip") < math.inf, f"pac_clip must be finite and positive, got {eff('pac_clip')}"),
        (eff("pac_zero_budget") and eff("no_privacy"), "pac_zero_budget and no_privacy exclude each other"),
        (("lora_r" in params or "lora_alpha" in params) and not eff("lora"), "lora_r and lora_alpha need lora"),
        ("ablation" in params and not eff("no_privacy"), "ablation needs no_privacy"),
        (("pac_adaptive_mi" in params or "pac_channel" in params) and eff("pac_zero_budget"),
         "pac_adaptive_mi and pac_channel have no effect under pac_zero_budget, which spends no MI budget"),
        (eff("pac_sort_by_len") and eff("train_as_classification"), "pac_sort_by_len would mis-pair the option groups of train_as_classification"),
        ("num_dev" in params and eff("trainer") == "none", "num_dev needs trainer zo, a run without training has no dev split"),
    ]
    for broken, message in rules:
        if broken:
            raise ConfigError(f"{where}: {message}")


def _resolve(line, where, path, settings, variants, bases):
    _mapping(line, where)
    for k in ("id", "variant"):
        if k not in line:
            raise ConfigError(f"{where}: missing key {k}")
    run_id = _check(line["id"], str, f"{where}: id")
    where = f"{path}: run {run_id}"
    vname = _check(line["variant"], str, f"{where}: variant")
    if vname not in variants:
        raise ConfigError(f"{where}: unknown variant {vname!r}")
    variant = variants[vname]
    base = bases[variant.base]
    layers = [(f"base {variant.base}", base.params)]
    sname = None
    if base.setting:
        if "setting" not in line:
            raise ConfigError(f"{where}: variant {vname} needs a setting")
        sname = _check(line["setting"], str, f"{where}: setting")
        if sname not in settings:
            raise ConfigError(f"{where}: unknown setting {sname!r}")
        layers.append((f"setting {sname}", settings[sname]))
    elif "setting" in line:
        raise ConfigError(f"{where}: variant {vname} takes no setting")
    layers.append((f"variant {vname}", variant.params))

    params, sources = {}, {}
    for source, layer in layers:
        for k, v in layer.items():
            if k in params:
                raise ConfigError(f"{where}: {k} is set by both {sources[k]} and {source}")
            params[k], sources[k] = v, source
    for k, v in _params({k: v for k, v in line.items() if k not in ("id", "variant", "setting")}, base.runner, where).items():
        if k in variant.params:
            raise ConfigError(f"{where}: {k} is fixed by variant {vname}")
        params[k], sources[k] = v, "run"

    missing = [k for k in base.requires + variant.requires if k not in params]
    if base.runner == "dpaggzo":
        missing += [k for k in KEYS["dpaggzo"] if k not in params and k not in missing]
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(missing)}")
    if base.runner == "paczo":
        _cross_checks(params, where)
    return Run(run_id, path, base.runner, sname, vname, params, sources)


def load(path):
    path = Path(path)
    settings, variants, bases = load_registry()
    lines = _yaml(path)
    if not isinstance(lines, list):
        raise ConfigError(f"{path}: expected a list of runs")
    runs = {}
    for i, line in enumerate(lines):
        run = _resolve(line, f"{path}: entry {i + 1}", path, settings, variants, bases)
        if run.id in runs:
            raise ConfigError(f"{path}: duplicate run id {run.id}")
        runs[run.id] = run
    return runs


def build(run):
    if run.runner == "dpaggzo":
        return DPAggZOEnv(**run.params)
    out = REPO / "result" / "runs" / run.id
    return PACZeroArguments(
        **run.params,
        output_dir=str(REPO / os.environ.get("PAC_CKPT_BASE", "checkpoints") / run.id),
        pac_diag_dir=str(out / "checkpoints"),
        result_file=str(out / "metrics.json"),
    )


def format_resolved(run, args):
    head = [
        f"config   {run.file}",
        f"run      {run.id}",
        f"runner   {run.runner}   variant {run.variant}" + (f"   setting {run.setting}" if run.setting else ""),
        "set by config",
    ]
    sources = dict(run.sources)
    if run.runner == "paczo":
        sources |= {k: "run id" for k in DERIVED}
        values = args.to_dict()
    else:
        values = dataclasses.asdict(args)
    width = max(map(len, values)) + 2
    head += [f"  {k:<{width}}{json.dumps(values[k]):<40} {s}" for k, s in sources.items()]
    head.append(f"resolved {type(args).__name__}")
    head += [f"  {k:<{width}}{json.dumps(v)}" for k, v in values.items()]
    return "\n".join(head)
