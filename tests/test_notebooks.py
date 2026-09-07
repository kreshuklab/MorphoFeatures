import importlib.util

import pytest


def test_notebook_collection_is_ordered_and_valid(repo_root):
    nbformat = pytest.importorskip("nbformat")
    paths = sorted((repo_root / "notebooks").glob("*.ipynb"))
    assert [path.name[:2] for path in paths] == ["01", "02", "03", "04"]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.cells
        assert notebook.cells[0].cell_type == "markdown"
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "/scratch/" not in source
        assert "label_id" in source
    real_source = "\n".join(cell.source for cell in nbformat.read(paths[-1], as_version=4).cells)
    assert "resolve_real_mae_config" in real_source
    assert "read_patch_qc_views" in real_source
    assert "SUBMIT_TO_SLURM = False" in real_source
    assert "label_id leakage" in real_source
    assert "nucleus-derived texture" in real_source


@pytest.mark.optional
@pytest.mark.slow
def test_cpu_mae_notebook_executes(repo_root, tmp_path, monkeypatch):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is not installed")
    nbformat = pytest.importorskip("nbformat")
    monkeypatch.setenv("MORPHOFEATURES_OUTPUT_ROOT", str(tmp_path / "notebook-outputs"))
    monkeypatch.delenv("MORPHOFEATURES_METRICS_PATH", raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(repo_root)
    path = repo_root / "notebooks" / "02_cpu_mae_training_and_encoding.ipynb"
    notebook = nbformat.read(path, as_version=4)
    namespace = {"display": lambda value: value}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path}:cell", "exec"), namespace)
    output = tmp_path / "notebook-outputs" / "notebooks" / "02_mae_smoke" / "embeddings.npy"
    assert output.is_file()
