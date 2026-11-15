"""Classification crops that keep the objects' real size (CropPlan mode 'fixed')."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from training_synth.datasets import formats, prepare, store
from training_synth.datasets.prepare import CROP_PAD, CropPlan, SplitPlan
from training_synth.evaluate import confused_pairs, package
from training_synth.runs.config import RunConfig, crop_rules
from training_synth.runs.manager import Manager
from training_synth.runs.store import Run
from training_synth.util import read_json, write_json

W, H = 1600, 1200  # render size; shrunk to 800 on the longest side: half the size
BG = 40


def _render(root: Path, split: str, name: str, rects: list[tuple[int, int, int, int, int]]) -> None:
    """One image with white rectangles (cls, x, y, w, h in pixels) and their YOLO box labels."""
    img = np.full((H, W, 3), BG, np.uint8)
    lines = []
    for cls, x, y, w, h in rects:
        img[y : y + h, x : x + w] = 255
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")


@pytest.fixture
def parts(tmp_path) -> Path:
    """Two classes of the same shape that differ only in size (like M3 and M4 screws)."""
    root = tmp_path / "parts"
    rng = random.Random(0)
    for split, n in (("train", 6), ("val", 2)):
        for i in range(n):
            x = 2 * rng.randint(50, 300)  # even: halving gives whole pixels
            _render(root, split, f"{split}{i}", [(0, x, 300, 60, 200), (1, x + 600, 500, 80, 266)])
    # one part at the image's corner, and one longer than a 128 px window after shrinking
    _render(root, "train", "edge", [(0, 0, 0, 60, 200), (1, 300, 300, 80, 600)])
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: {0: small, 1: large}\n"
    )
    return root


def _white_size(crop: np.ndarray) -> tuple[int, int]:
    m = crop.min(axis=2) > 200
    cols, rows = np.where(m.any(0))[0], np.where(m.any(1))[0]
    return int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


def _crops(out: Path) -> list[Path]:
    return sorted(out.glob("*/*/*.png"))


PLAN = CropPlan(mode="fixed", size_px=128, longest_side_px=800)


def test_fixed_crops_are_one_size_and_keep_real_sizes(parts, tmp_path):
    ds = formats.load(parts / "data.yaml")
    out = tmp_path / "run"
    m = prepare.prepare(ds, out, "classify", crop=PLAN)
    crops = _crops(out)
    assert crops and all(p.suffix == ".png" for p in crops)  # lossless
    for p in crops:
        assert cv2.imread(str(p)).shape == (128, 128, 3)  # whatever the object's size
    # the same scale for every crop: half the render's pixels (1600 -> 800)
    for p in (out / "train" / "small").glob("train*.png"):
        assert _white_size(cv2.imread(str(p))) == (30, 100)
    for p in (out / "train" / "large").glob("train*.png"):
        assert _white_size(cv2.imread(str(p))) == (40, 128)  # 133 px high: cut by the window
    assert m["crops"]["mode"] == "fixed" and m["crops"]["card"] == PLAN.card()
    assert m["crops"]["crops"] == {"small": 9, "large": 9}
    # every 'large' is 133 px high after shrinking: larger than the 128 px window
    assert m["crops"]["larger_than_window"] == {"large": {"crops": 9, "share": 1.0}}


def test_two_parts_of_one_render_keep_their_width_ratio(parts, tmp_path):
    ds = formats.load(parts / "data.yaml")
    out = tmp_path / "run"
    prepare.prepare(ds, out, "classify", crop=CropPlan(size_px=256, longest_side_px=800))
    small = _white_size(cv2.imread(str(out / "train" / "small" / "train0_0.png")))
    large = _white_size(cv2.imread(str(out / "train" / "large" / "train0_1.png")))
    assert large[0] / small[0] == pytest.approx(80 / 60, abs=0.04)
    assert small == (30, 100) and large == (40, 133)


def test_a_part_at_the_edge_is_padded_with_the_pad_colour(parts, tmp_path):
    ds = formats.load(parts / "data.yaml")
    out = tmp_path / "run"
    prepare.prepare(ds, out, "classify", crop=PLAN)
    crop = cv2.imread(str(out / "train" / "small" / "edge_0.png"))
    assert crop.shape == (128, 128, 3)
    # the box centre is (15, 50) in the shrunk image: 49 columns and 14 rows lie outside it
    assert (crop[:, :40] == CROP_PAD).all() and (crop[:10, :] == CROP_PAD).all()
    assert _white_size(crop) == (30, 100)


def test_scaled_mode_is_the_old_crop(parts, tmp_path):
    ds = formats.load(parts / "data.yaml")
    out = tmp_path / "run"
    m = prepare.prepare(ds, out, "classify", crop=CropPlan(mode="scaled"))
    crops = sorted(out.glob("*/*/*.jpg"))
    assert crops and not _crops(out)
    side = {cv2.imread(str(p)).shape[0] for p in (out / "train" / "small").glob("train*.jpg")}
    assert side == {240}  # 200 px x (1 + 2 x 0.1): the window grows with the part
    assert m["crops"] == {"mode": "scaled", "padding": 0.1}


def test_folder_datasets_are_not_cropped(folders_ds, tmp_path):
    ds = formats.load(folders_ds)
    out = tmp_path / "run"
    m = prepare.prepare(ds, out, "classify", split=SplitPlan(mode="resplit"), crop=PLAN)
    files = [p for p in out.glob("*/*/*") if p.is_file()]
    sources = {p.suffix for p in folders_ds.rglob("*") if p.is_file()}
    assert files and {p.suffix for p in files} <= sources  # the images as they are
    assert m["crops"] is None


def test_fixed_crop_rules_and_old_runs():
    cfg = RunConfig(name="c", project="p", dataset="d", task="classify")
    a = cfg.train_args()
    assert cfg.crop["mode"] == "fixed" and a["imgsz"] == 256
    assert a["scale"] == 0 and a["auto_augment"] is None and a["erasing"] <= 0.2
    _, notes = crop_rules(cfg.train_args(apply_crop=False), "classify", cfg.crop)
    assert any("auto-augment" in n for n in notes) and any("image size" in n for n in notes)
    cfg.train = {"imgsz": 640, "erasing": 0.1, "hsv_h": 0.02}
    a = cfg.train_args()
    assert a["imgsz"] == 256 and a["erasing"] == 0.1 and a["hsv_h"] == 0.02  # colour is kept
    # a run.yaml from before fixed crops: the old crops and settings, unchanged
    old = cfg.to_dict()
    del old["crop"]
    again = RunConfig.from_dict(old)
    assert again.crop == {"mode": "scaled"} and again.train_args()["imgsz"] == 640
    # detection is never touched; folder datasets are not cropped
    det = RunConfig(name="d", project="p", dataset="d", task="detect", train={"imgsz": 1024})
    assert det.train_args()["imgsz"] == 1024
    cfg.set_label_kind("classes")
    assert cfg.crop == {"mode": "none"} and cfg.train_args()["imgsz"] == 640
    bad = RunConfig(name="c", project="p", dataset="d", task="classify", crop={"mode": "x"})
    assert any("crop" in p for p in bad.problems())


def test_fixed_training_transform_does_not_stretch(parts, tmp_path):
    """Ultralytics' classification augmentation, with a fixed run's settings, keeps every width.

    Its RandomResizedCrop stretches by up to +-15 % per side unless scale is 0 and the crop is
    square (then the whole crop is the only choice), and RandAugment shears: crop_rules turns
    both off. A square crop must come out with the same object size in every sample.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data import ClassificationDataset

    ds = formats.load(parts / "data.yaml")
    out = tmp_path / "run"
    prepare.prepare(ds, out, "classify", crop=PLAN)
    cfg = RunConfig(name="c", project="p", dataset="d", task="classify", crop=vars(PLAN))
    cfg.train = {"erasing": 0}  # erasing blanks patches (no stretch) and would spoil the measure
    args = {k: v for k, v in cfg.train_args().items() if k in get_cfg().__dict__}
    data = ClassificationDataset(str(out / "train"), get_cfg(overrides=args), augment=True)
    idx = next(i for i, s in enumerate(data.samples) if Path(s[0]).name == "train0_0.png")
    random.seed(0)
    sizes = set()
    for _ in range(40):
        t = data[idx]["img"]
        assert tuple(t.shape[1:]) == (128, 128)  # never resized
        g = t.mean(0).numpy()
        m = g > (g.max() + g.min()) / 2
        cols, rows = np.where(m.any(0))[0], np.where(m.any(1))[0]
        sizes.add((int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)))
    assert sizes == {(30, 100)}
    # the check is sharp: Ultralytics' defaults (zoom 0.5, RandAugment) do change the size
    plain = ClassificationDataset(
        str(out / "train"), get_cfg(overrides={"imgsz": 128}), augment=True
    )
    changed = set()
    for _ in range(40):
        g = plain[idx]["img"].mean(0).numpy()
        m = g > (g.max() + g.min()) / 2
        cols = np.where(m.any(0))[0]
        if len(cols):
            changed.add(int(cols.max() - cols.min() + 1))
    assert changed != {30}


def test_model_card_describes_fixed_crops_only(workspace, parts, folders_ds):
    proj = workspace.project("default")
    store.import_dataset(proj, parts / "data.yaml", name="parts")
    store.import_dataset(proj, folders_ds, name="folders")
    mgr = Manager(workspace)

    def card(dataset: str, crop: dict) -> dict:
        cfg = RunConfig(
            name=dataset,
            project="default",
            dataset=dataset,
            task="classify",
            crop=dict(crop),
            split={"mode": "resplit"},
        )
        run = mgr.create(proj, cfg)
        ds, _ = store.open_dataset(proj, dataset)
        plan = run.config.crop
        prepare.prepare(
            ds,
            run.data_dir,
            "classify",
            split=SplitPlan(mode="resplit"),
            crop=CropPlan.from_dict(plan) if plan["mode"] != "none" else None,
        )
        run.weights_dir.mkdir(parents=True, exist_ok=True)
        (run.weights_dir / "best.pt").write_bytes(b"weights")
        res = {"confused_pairs": [{"a": "small", "b": "large", "a_as_b": 1}]}
        return read_json(package(run, res) / "model_card.json"), run

    fixed, run = card("parts", {"mode": "fixed", "size_px": 128, "longest_side_px": 800})
    assert fixed["classifier_crop"] == {
        "version": 1,
        "mode": "fixed",
        "longest_side_px": 800,
        "size_px": 128,
        "pad": [114, 114, 114],
        "interpolation": "area",
        "centre": "box",
        "labels": "class",
    }
    assert fixed["imgsz"] == 128 and fixed["confused_pairs"][0]["a"] == "small"
    scaled, _ = card("parts", {"mode": "scaled"})
    assert "classifier_crop" not in scaled
    folders, frun = card("folders", {"mode": "fixed"})
    assert "classifier_crop" not in folders and frun.config.crop == {"mode": "none"}
    # a package made again after the crops changed loses the old block
    m = read_json(run.data_dir / "manifest.json")
    m["crops"] = {"mode": "scaled", "padding": 0.1}
    write_json(run.data_dir / "manifest.json", m)
    assert "classifier_crop" not in read_json(package(run, {}) / "model_card.json")


def test_confused_pairs():
    # matrix[predicted][true]: 10 a (2 called b), 10 b (5 called a), 4 c (all right)
    mat = [[8, 5, 0], [2, 5, 0], [0, 0, 4]]
    pairs = confused_pairs(mat, ["a", "b", "c"])
    assert pairs == [
        {
            "a": "a",
            "b": "b",
            "a_as_b": 2,
            "b_as_a": 5,
            "a_images": 10,
            "b_images": 10,
            "rate": 0.35,
        }
    ]
    assert confused_pairs([[3, 0], [0, 3]], ["a", "b"]) == []


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_wizard_shows_crop_settings_for_classify(tmp_path, workspace, yolo_seg):
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation

    store._CACHE.clear()
    store.import_dataset(workspace.project("default"), yolo_seg, name="shapes")
    main = tmp_path / "main.py"
    main.write_text(
        "from training_synth.gui.app import run_app\n"
        "from training_synth.workspace import Workspace\n"
        f"run_app(Workspace(r'{workspace.root}'), show=False)\n"
    )
    async with user_simulation(main_file=main) as user:
        await user.open("/new?dataset=shapes")
        await user.should_see("What should the model do?")
        next(iter(user.find(kind=ui.radio, marker="task").elements)).set_value("classify")
        await user.should_see("Object crops")
        await user.should_see("Crop size (px)")
        await user.should_see("Shrink images to (px, longest side)")
        await user.should_see("Fixed crops: zoom and auto-augment are off")
        await user.should_see("fixed crops set:")
        user.find(marker="name").type("-crops")
        user.find("Save without starting").click()
        await user.should_see("Results appear when training ends", retries=100)
    cfg = Run(workspace.project("default").runs()[0]).config
    assert cfg.task == "classify" and cfg.crop["mode"] == "fixed"
    assert cfg.train_args()["imgsz"] == 256 and cfg.train_args()["auto_augment"] is None


def test_classify_metrics_get_class_names_from_the_confusion_matrix():
    """Ultralytics' classification metrics have no names; its confusion matrix has them."""
    from types import SimpleNamespace

    from training_synth.evaluate import metrics_dict

    cm = SimpleNamespace(matrix=np.array([[8.0, 5.0], [2.0, 5.0]]), names={0: "M3", 1: "M4"})
    m = SimpleNamespace(results_dict={"metrics/accuracy_top1": 0.65}, confusion_matrix=cm, speed={})
    res = metrics_dict(m, "classify")
    assert res["confusion"]["names"] == ["M3", "M4"]
    assert res["confused_pairs"][0]["a"] == "M3" and res["confused_pairs"][0]["b_as_a"] == 5
