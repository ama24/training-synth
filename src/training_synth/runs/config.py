"""A run's settings (run.yaml): everything needed to reproduce it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import params as P
from ..util import read_yaml, write_yaml


@dataclass
class ModelChoice:
    family: str = "yolo26"
    size: str = "n"
    start: str = "pretrained"  # pretrained | scratch | run | file
    source: str = ""  # run id (start=run) or a weights path (start=file)


@dataclass
class RunOptions:
    snapshot_every: int = 10  # also keep a copy of the weights every N epochs (0 = off)
    keep_snapshots: int = 5
    device: str = ""  # "" = first GPU if there is one, else CPU; "cpu"; "0"; "0,1"
    watch_images: int = 8  # validation images shown with predictions after each epoch
    notify: bool = True
    executor: str = "local"  # local | ssh:<remote name>


@dataclass
class RunConfig:
    name: str
    project: str
    dataset: str
    task: str
    model: ModelChoice = field(default_factory=ModelChoice)
    classes: dict[str, Any] = field(default_factory=dict)  # ClassPlan: drop / merge / rename
    split: dict[str, Any] = field(default_factory=lambda: {"mode": "keep"})  # SplitPlan
    subset: float = 1.0
    balance: str = "none"  # none | oversample
    aug_preset: str = "standard"
    train: dict[str, Any] = field(default_factory=dict)  # Ultralytics arguments (see params.py)
    options: RunOptions = field(default_factory=RunOptions)
    notes: str = ""
    # Which defaults the settings were saved against (see params.OLD_DEFAULTS). A run.yaml
    # without it is from before version 2.
    defaults_version: int = P.DEFAULTS_VERSION
    # Classification crops from box/mask labels (datasets.prepare.CropPlan): mode fixed | scaled,
    # or none for folder-per-class datasets, which are not cropped (see set_label_kind)
    crop: dict[str, Any] = field(default_factory=lambda: {"mode": "fixed"})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunConfig:
        d = dict(d)
        d.setdefault("defaults_version", 1)
        d.setdefault("crop", {"mode": "scaled"})  # saved before fixed crops existed
        model = ModelChoice(**(d.pop("model", None) or {}))
        options = RunOptions(**(d.pop("options", None) or {}))
        known = set(cls.__dataclass_fields__)
        return cls(model=model, options=options, **{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(read_yaml(path))

    def save(self, path: Path) -> None:
        write_yaml(path, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def train_args(self, apply_crop: bool = True) -> dict[str, Any]:
        """Ultralytics arguments: defaults < augmentation preset < explicit values < crop rules."""
        args = P.defaults(self.task)
        old = {k: v for k, v in P.old_defaults(self.defaults_version).items() if k in args}
        args.update(old)
        preset = dict(P.AUG_PRESETS.get(self.aug_preset, {}))
        if self.aug_preset == "standard":  # 'standard' is the defaults of its time
            preset.update(old)
        args.update({k: v for k, v in preset.items() if k in args})
        clean, _ = P.validate(self.train, self.task)
        args.update(clean)
        return crop_rules(args, self.task, self.crop)[0] if apply_crop else args

    def set_label_kind(self, label_kind: str) -> None:
        """Folder-per-class datasets have no boxes to crop: their crop mode is none."""
        if label_kind == "classes":
            self.crop = {"mode": "none"}
        elif self.crop.get("mode") == "none":
            self.crop = {"mode": "fixed"}

    def fixed_crops(self) -> bool:
        return self.task == "classify" and self.crop.get("mode") == "fixed"

    def problems(self) -> list[str]:
        from ..models import FAMILIES

        out = []
        if self.model.family not in FAMILIES:
            out.append(f"unknown model family {self.model.family!r}")
        elif self.task not in FAMILIES[self.model.family].tasks:
            out.append(f"{FAMILIES[self.model.family].label} cannot do {self.task}")
        elif self.model.size not in FAMILIES[self.model.family].sizes:
            out.append(f"{FAMILIES[self.model.family].label} has no size {self.model.size!r}")
        if self.model.start in ("run", "file") and not self.model.source:
            out.append("choose the run or weights file to start from")
        _, p = P.validate(self.train, self.task)
        if self.crop.get("mode") != "none":
            from ..datasets.prepare import CropPlan

            try:
                CropPlan.from_dict(self.crop)
            except ValueError as exc:
                out.append(f"crop: {exc}")
        return out + p


FIXED_CROP_MAX_ERASING = 0.2


def crop_rules(
    args: dict[str, Any], task: str, crop: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[str]]:
    """Training arguments for fixed classification crops, and what was changed and why.

    Fixed crops carry the object's size in their pixels, so nothing may resize or stretch them:
    - imgsz = the crop size: Ultralytics never resizes the crop;
    - scale 0: its RandomResizedCrop then has to keep the whole (square) crop, which also turns
      off its aspect-ratio stretch (+-15 % per side);
    - auto_augment off: RandAugment shears and shifts, which changes widths;
    - erasing at most 0.2: blanked patches hide the edges that show the size.
    Colour and flips stay: they do not change sizes.
    """
    if task != "classify" or not crop or crop.get("mode") != "fixed":
        return args, []
    from ..datasets.prepare import CropPlan

    plan = CropPlan.from_dict(crop)
    out = dict(args)
    notes = []
    if int(out.get("imgsz") or 0) != plan.size_px:
        notes.append(f"image size {out.get('imgsz')} -> {plan.size_px} px (the crop size)")
    out["imgsz"] = plan.size_px
    if float(out.get("scale") or 0) != 0:
        notes.append(f"zoom (scale) {out.get('scale')} -> 0")
    out["scale"] = 0.0
    if out.get("auto_augment"):
        notes.append(f"auto-augment {out['auto_augment']} -> off (it shears)")
    out["auto_augment"] = None
    if float(out.get("erasing") or 0) > FIXED_CROP_MAX_ERASING:
        notes.append(f"random erasing {out['erasing']} -> {FIXED_CROP_MAX_ERASING}")
        out["erasing"] = FIXED_CROP_MAX_ERASING
    return out, notes
