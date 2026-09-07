"""Normalize the maintained real-data tutorial without duplicating its narrative."""

from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "04_real_platynereis_mae_workflow.ipynb"


def build(path: Path = NOTEBOOK) -> Path:
    """Clear execution state while preserving the canonical pedagogical cells.

    The real tutorial changes with audited data and model contracts. Keeping a
    second copy of every cell inside this script previously allowed an obsolete
    single-patch workflow to overwrite the grouped-nucleus notebook.
    """

    destination = Path(path)
    notebook = nbformat.read(destination, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    return destination


if __name__ == "__main__":
    build()
