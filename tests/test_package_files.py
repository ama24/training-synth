"""A dataset's `package_files` (e.g. screw-synth's classes.yaml and parts.json) reach the model
package: dataset yaml -> run dataset -> package/ next to best.pt, listed in model_card.json."""

import shutil
import zipfile

import pytest

from conftest import make_yolo
from training_synth.datasets import formats, prepare
from training_synth.evaluate import package
from training_synth.export import package_zip
from training_synth.runs.config import RunConfig
from training_synth.runs.store import Run
from training_synth.util import write_json
from training_synth.workspace import default_root


def test_package_files_travel_into_the_model_package(tmp_path, workspace):
    root = make_yolo(tmp_path / "shapes")
    (root / "classes.yaml").write_text("names: {0: a, 1: b, 2: c}\n")
    (root / "parts.json").write_text('{"format": "screwsynth-parts"}\n')
    with (root / "data.yaml").open("a") as f:
        f.write("package_files: [classes.yaml, parts.json, missing.json, ../../etc/passwd]\n")
    ds = formats.load(root / "data.yaml")
    assert sorted(ds.package_files) == ["classes.yaml", "parts.json"]  # missing / outside: left out

    cfg = RunConfig(name="r", project="default", dataset="shapes", task="segment")
    run = Run.create(workspace.project("default"), cfg)
    m = prepare.prepare(ds, run.data_dir, "segment", hash_files=False)
    assert m["package_files"] == ["classes.yaml", "parts.json"]
    run.weights_dir.mkdir(parents=True)
    (run.weights_dir / "best.pt").write_bytes(b"weights")
    (root / "parts.json").write_text("changed later\n")  # the run keeps the copy it trained with
    d = package(run, {"mean": {}})
    card = (d / "model_card.json").read_text()
    assert '"package_files": [\n    "classes.yaml",\n    "parts.json"\n  ]' in card
    assert (d / "classes.yaml").read_text().startswith("names:")
    assert "screwsynth-parts" in (d / "parts.json").read_text()


def _exported_run(workspace, name="My run", precision="fp32"):
    """A run with a package and an NCNN export, as export_model leaves them on disk."""
    run = Run.create(
        workspace.project("default"),
        RunConfig(name=name, project="default", dataset="d", task="segment"),
    )
    d = run.package_dir
    work = d / "exports" / f"ncnn_{precision}"
    model = work / "best_ncnn_model"
    (model / "__pycache__").mkdir(parents=True)
    (work / "val").mkdir()
    for f in ("metadata.yaml", "model.ncnn.bin", "model.ncnn.param", "model_ncnn.py"):
        (model / f).write_text(f)
    (model / "__pycache__" / "model_pnnx.cpython-312.pyc").write_text("x")
    (work / "val" / "BoxF1_curve.png").write_text("x")
    (d / "best.pt").write_text("w")
    (d / "classes.yaml").write_text("names: {}\n")
    (d / "parts.json").write_text("{}")
    # the path as recorded on another computer: the zip must not depend on it
    rec = {"format": "ncnn", "precision": precision, "size_mb": 1, "path": r"C:\x\best_ncnn_model"}
    write_json(d / "exports.json", [rec])
    write_json(d / "model_card.json", {"package_files": ["classes.yaml", "parts.json", "gone.txt"]})
    return run


def test_package_zip_holds_what_another_program_needs(workspace):
    run = _exported_run(workspace)
    out = package_zip(run, "ncnn")
    assert out == run.package_dir / "My_run_ncnn.zip"
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert sorted(z.namelist()) == [
            "classes.yaml",
            "exports.json",
            "exports/ncnn_fp32/best_ncnn_model/metadata.yaml",
            "exports/ncnn_fp32/best_ncnn_model/model.ncnn.bin",
            "exports/ncnn_fp32/best_ncnn_model/model.ncnn.param",
            "model_card.json",
            "parts.json",
        ]
    assert [p.name for p in run.package_dir.glob(".*")] == []  # no temporary file left behind


def test_package_zip_names_a_lower_precision(workspace):
    run = _exported_run(workspace, precision="fp16")
    assert package_zip(run, "ncnn", "fp16").name == "My_run_ncnn_fp16.zip"


def test_package_zip_of_a_missing_export_says_so(workspace):
    run = _exported_run(workspace)
    with pytest.raises(ValueError, match="no onnx fp32 export"):
        package_zip(run, "onnx")
    shutil.rmtree(run.package_dir / "exports")
    with pytest.raises(FileNotFoundError, match="export is missing"):
        package_zip(run, "ncnn")


def test_a_dataset_without_package_files_packages_as_before(tmp_path, workspace):
    ds = formats.load(make_yolo(tmp_path / "plain") / "data.yaml")
    assert ds.package_files == {}
    run = Run.create(
        workspace.project("default"),
        RunConfig(name="p", project="default", dataset="plain", task="detect"),
    )
    prepare.prepare(ds, run.data_dir, "detect", hash_files=False)
    run.weights_dir.mkdir(parents=True)
    (run.weights_dir / "best.pt").write_bytes(b"w")
    assert sorted(p.name for p in package(run, {"mean": {}}).iterdir()) == [
        "best.pt",
        "model_card.json",
    ]


def test_the_workspace_from_before_the_rename_is_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAINING_SYNTH_HOME", raising=False)
    monkeypatch.delenv("MTRAIN_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert default_root() == tmp_path / "training-synth-workspace"
    (tmp_path / "model-trainer-workspace").mkdir()
    assert default_root() == tmp_path / "model-trainer-workspace"  # runs stay where they are
    monkeypatch.setenv("MTRAIN_HOME", str(tmp_path / "old_env"))
    assert default_root() == tmp_path / "old_env"
    monkeypatch.setenv("TRAINING_SYNTH_HOME", str(tmp_path / "new_env"))
    assert default_root() == tmp_path / "new_env"
