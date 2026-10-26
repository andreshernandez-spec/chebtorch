"""The vendoring contract (PROJECT.md section 3), checked for every family.

A family is its module plus the sibling modules it imports: its generated
tables (_<family>_data.py) and the shared helpers (cheb_common.py). Together they
import only math, torch and each other; the data is exactly what its
provenance hash names (and, when chebax is importable, exactly chebax's
tables); and the files copied alone into another package, the family module
under another name, work with chebtorch itself unimportable.
"""

import ast
import importlib
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "chebtorch"
sys.path.insert(0, str(ROOT / "tools"))
from export import FAMILIES, data_sha256  # noqa: E402


def _imports(path):
    """(top-level absolute imports, sibling modules) of one source file."""
    absolute, siblings = set(), set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            absolute |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                absolute.add(node.module.split(".")[0])
            else:
                assert node.level == 1, f"{path.name}: import from outside the package"
                siblings |= {node.module} if node.module else {a.name for a in node.names}
    return absolute, siblings


def family_files(family):
    """The family module and every sibling module it reaches, by name."""
    files, todo = {}, [family]
    while todo:
        name = todo.pop()
        if name not in files:
            files[name] = SRC / f"{name}.py"
            todo += sorted(_imports(files[name])[1])
    return files


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_imports_only_math_torch_and_each_other(family):
    for name, path in family_files(family).items():
        absolute, _ = _imports(path)
        assert absolute <= {"math", "torch"}, (name, absolute)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_docstring_names_the_files_to_copy(family):
    doc = importlib.import_module(f"chebtorch.{family}").__doc__
    for name in family_files(family):
        assert name == family or f"{name}.py" in doc, name


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_data_matches_its_provenance(family):
    mod = importlib.import_module(f"chebtorch._{family}_data")
    assert data_sha256(mod.SHAPES, mod.DATA) == mod.PROVENANCE["data_sha256"]
    # an export from uncommitted chebax tables names no commit anyone can check
    assert not mod.PROVENANCE["chebax_commit"].endswith("-dirty")


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_data_is_chebax_tables(family):
    np = pytest.importorskip("numpy")
    pytest.importorskip("chebax")
    tm = importlib.import_module(FAMILIES[family][0])
    mod = importlib.import_module(f"chebtorch._{family}_data")
    for name, values in mod.DATA.items():
        ref = np.asarray(getattr(tm, name), dtype=np.float64)
        assert ref.shape == mod.SHAPES[name]
        assert np.array_equal(ref.ravel(), np.asarray(values)), name
    for name in FAMILIES[family][1]:
        assert getattr(mod, name) == getattr(tm, name), name


COPY_CHECK = r"""
import json, sys
sys.modules["chebtorch"] = None            # the copy must not need the library
import torch
from hostpkg.losses import tweedie_loss as m
fx = json.load(open(sys.argv[1]))
worst = 0.0
for c in fx["log_wright_bessel0"]:
    v = m.log_wright_bessel0(c["rho"], torch.tensor(c["z"], dtype=torch.float64)).item()
    worst = max(worst, abs(v - c["value"]) / max(1.0, abs(c["value"])))
for c in fx["tweedie_logpdf"]:
    v = m.tweedie_logpdf(c["p"], c["mu"], c["phi"], torch.tensor(c["y"], dtype=torch.float64)).item()
    worst = max(worst, abs(v - c["value"]) / max(1.0, abs(c["value"])))
print(worst)
"""


def test_copied_files_work_alone(tmp_path):
    # the family module under another name, its siblings under their own, in
    # a package that knows nothing of chebtorch
    pkg = tmp_path / "hostpkg" / "losses"
    pkg.mkdir(parents=True)
    (tmp_path / "hostpkg" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    for name, path in family_files("tweedie").items():
        shutil.copy(path, pkg / ("tweedie_loss.py" if name == "tweedie" else path.name))
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", COPY_CHECK,
                          str(ROOT / "tests" / "fixtures" / "tweedie.json")],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout) <= 3e-12            # the float64 density bar
