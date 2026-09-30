import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fresh_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    subprocess.run(
        [
            sys.executable, "scripts/export_web.py",
            "--out_dir", str(out / "models"),
            "--reference", str(out / "reference.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    return out


def assert_close(got, want, path="reference"):
    # Torch builds differ by about 1e-6 in float32, so use the javascript tests' 1e-4 tolerance.
    if isinstance(got, float) or isinstance(want, float):
        assert math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-4), f"{path}: {got} != {want}"
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            assert_close(g, w, f"{path}[{i}]")
    elif isinstance(want, dict):
        assert got.keys() == want.keys(), f"{path}: keys differ"
        for key in want:
            assert_close(got[key], want[key], f"{path}.{key}")
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def test_committed_web_models_match_the_checkpoints(fresh_export):
    committed = ROOT / "web" / "models"
    fresh = fresh_export / "models"
    assert sorted(p.name for p in fresh.iterdir()) == sorted(p.name for p in committed.iterdir())
    for path in fresh.iterdir():
        stale = f"web/models/{path.name} is stale, rerun scripts/export_web.py"
        assert path.read_bytes() == (committed / path.name).read_bytes(), stale


def test_committed_reference_matches_pytorch(fresh_export):
    committed = json.loads((ROOT / "tests" / "web" / "reference.json").read_text())
    fresh = json.loads((fresh_export / "reference.json").read_text())
    assert_close(committed, fresh)
