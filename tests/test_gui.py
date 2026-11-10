"""Simulated user (NiceGUI's test client): the pages build and the main flows work without a browser."""

from __future__ import annotations

import pytest
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from training_synth.datasets import store
from training_synth.runs.store import Run


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def app_main(tmp_path, workspace, yolo_seg):
    store._CACHE.clear()
    store.import_dataset(workspace.project("default"), yolo_seg, name="shapes")
    main = tmp_path / "main.py"
    main.write_text(
        "from training_synth.gui.app import run_app\n"
        "from training_synth.workspace import Workspace\n"
        f"run_app(Workspace(r'{workspace.root}'), show=False)\n"
    )
    return main


@pytest.mark.anyio
async def test_pages_open(app_main):
    async with user_simulation(main_file=app_main) as user:
        await user.open("/")
        await user.should_see("Import a dataset")
        await user.open("/datasets")
        await user.should_see("Click a dataset to see")
        await user.open("/glossary")
        await user.should_see("mAP50-95")
        await user.open("/settings")
        await user.should_see("Phone notifications")
        await user.open("/runs")
        await user.should_see("No runs yet")


@pytest.mark.anyio
async def test_dataset_health_report(app_main):
    async with user_simulation(main_file=app_main) as user:
        await user.open("/dataset/shapes")
        await user.should_see("Health report")
        await user.should_see("Class balance", retries=100)
        await user.should_see("Recommended training image size", retries=20)


@pytest.mark.anyio
async def test_wizard_creates_a_run(app_main, workspace):
    async with user_simulation(main_file=app_main) as user:
        await user.open("/new?dataset=shapes")
        await user.should_see("What should the model do?")
        await user.should_see("Segmentation (masks)")
        user.find(marker="name").type("-test")
        user.find("Save without starting").click()
        await user.should_see("Results appear when training ends", retries=100)
    runs = workspace.project("default").runs()
    assert len(runs) == 1
    r = Run(runs[0])
    assert r.config.dataset == "shapes" and r.config.task == "segment"
    assert r.state()["state"] == "created"


@pytest.mark.anyio
async def test_run_page_of_created_run(app_main, workspace):
    from training_synth.runs.config import RunConfig
    from training_synth.runs.manager import Manager

    run = Manager(workspace).create(
        workspace.project("default"),
        RunConfig(name="r1", project="default", dataset="shapes", task="detect"),
    )
    async with user_simulation(main_file=app_main) as user:
        await user.open(f"/run/{run.id}")
        await user.should_see("r1")
        await user.should_see("Results appear when training ends")
        user.find(kind=ui.button, content="Copy & change").click()
        await user.should_see("What should the model do?")


@pytest.mark.anyio
async def test_create_zip_button(app_main, workspace):
    from training_synth.runs.config import RunConfig
    from training_synth.runs.manager import Manager
    from training_synth.util import write_json

    run = Manager(workspace).create(
        workspace.project("default"),
        RunConfig(name="r1", project="default", dataset="shapes", task="detect"),
    )
    run.weights_dir.mkdir(parents=True)
    (run.weights_dir / "best.pt").write_bytes(b"w")
    model = run.package_dir / "exports" / "onnx_fp32" / "best.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    rec = {"format": "onnx", "precision": "fp32", "size_mb": 1, "path": str(model)}
    write_json(run.package_dir / "exports.json", [rec])
    write_json(run.package_dir / "model_card.json", {})
    run.set_state("done")
    async with user_simulation(main_file=app_main) as user:
        await user.open(f"/run/{run.id}")
        await user.should_see(marker="zip-onnx-fp32", retries=50)
        user.find(marker="zip-onnx-fp32").click()
        await user.should_see("Created", retries=100)
    assert (run.package_dir / "r1_onnx.zip").is_file()


@pytest.mark.anyio
async def test_wizard_class_table_and_bulk_drop(app_main):
    async with user_simulation(main_file=app_main) as user:
        await user.open("/new?dataset=shapes")
        await user.should_see("Merge into / new name")
        await user.should_see("triangle")
        await user.should_see("3 classes")
        user.find("Drop classes with fewer objects").click()
        await user.should_see("dropped")


@pytest.mark.anyio
async def test_import_asks_which_dataset(app_main, tmp_path):
    from conftest import make_yolo

    src = make_yolo(tmp_path / "multi", groups=False)
    (src / "other.yaml").write_text((src / "data.yaml").read_text())
    (src / "data.yaml").rename(src / "first.yaml")
    async with user_simulation(main_file=app_main) as user:
        await user.open("/datasets")
        user.find(marker="import").click()
        user.find(marker="path").type(str(src))
        user.find(marker="do-import").click()
        await user.should_see("This folder holds 2 datasets", retries=50)
        user.find(marker="pick-other").click()
        await user.should_see("Health report", retries=50)
