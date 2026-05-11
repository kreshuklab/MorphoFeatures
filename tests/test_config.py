"""Tests for configuration loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from morphofeatures.config.loading import load_config, merge_dicts


class ConfigTests(unittest.TestCase):
    """Validate YAML loading and merging."""

    def test_merge_dicts_is_recursive_and_non_mutating(self) -> None:
        """Nested dictionaries are merged without mutating inputs."""

        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 4}}
        merged = merge_dicts(base, override)
        self.assertEqual(merged, {"a": {"x": 1, "y": 4}, "b": 3})
        self.assertEqual(base["a"]["y"], 2)

    def test_load_config_with_defaults(self) -> None:
        """Run-specific config overrides defaults."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            defaults = tmp_path / "defaults.yaml"
            config = tmp_path / "config.yaml"
            defaults.write_text("model:\n  name: DeepGCN\n  channels: 64\n", encoding="utf-8")
            config.write_text("model:\n  channels: 32\n", encoding="utf-8")
            loaded = load_config(config, defaults)
        self.assertEqual(loaded["model"]["name"], "DeepGCN")
        self.assertEqual(loaded["model"]["channels"], 32)


if __name__ == "__main__":
    unittest.main()
