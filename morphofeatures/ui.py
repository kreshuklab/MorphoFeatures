"""Launch the Streamlit workflow workspace with a dependency-light fallback."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import List, Tuple


@dataclass(frozen=True)
class NavigationItem:
    stage: str
    command: str
    documentation: str


NAVIGATION: Tuple[NavigationItem, ...] = (
    NavigationItem("Validate bundled data", "morphofeatures validate", "docs/legacy_reproduction.md"),
    NavigationItem(
        "Run class prediction",
        "morphofeatures classify --folds 5",
        "docs/analysis_and_mobie.md",
    ),
    NavigationItem(
        "Project and cluster",
        "morphofeatures project --subset 500 --cluster-method kmeans",
        "docs/analysis_and_mobie.md",
    ),
    NavigationItem(
        "Create synthetic data",
        "morphofeatures synthetic outputs/synthetic",
        "docs/data_preparation.md",
    ),
    NavigationItem(
        "Train a 3D MAE",
        "morphofeatures mae-train --config configs/smoke.yaml",
        "docs/modern_mae_workflow.md",
    ),
)


def print_navigation() -> None:
    print("MorphoFeatures workflow navigator")
    print("=" * 34)
    for index, item in enumerate(NAVIGATION, start=1):
        print("{}. {}\n   {}\n   {}".format(index, item.stage, item.command, item.documentation))


def dashboard_path() -> Path:
    return Path(__file__).resolve().with_name("dashboard.py")


def build_streamlit_command(
    port: int = 1010, address: str = "localhost", headless: bool = False
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path()),
        "--server.address",
        address,
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "false",
    ]


def run_navigator(port: int = 8501, address: str = "localhost", headless: bool = False) -> int:
    if importlib.util.find_spec("streamlit") is None:
        print_navigation()
        print("\nInstall morphofeatures[ui] to open the Streamlit workspace.")
        return 1
    return subprocess.call(build_streamlit_command(port=port, address=address, headless=headless))
