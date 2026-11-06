"""A dataset's `package_files` (e.g. screw-synth's classes.yaml and parts.json) reach the model
package: dataset yaml -> run dataset -> package/ next to best.pt, listed in model_card.json."""

from conftest import make_yolo
from training_synth.datasets import formats, prepare
from training_synth.evaluate import package
from training_synth.runs.config import RunConfig
from training_synth.runs.store import Run
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
