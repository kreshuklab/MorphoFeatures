from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).resolve().parents[1]
