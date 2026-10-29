import importlib

import pytest


@pytest.mark.parametrize("package", ["core", "tools", "integrations", "evals"])
def test_package_imports(package: str) -> None:
    assert importlib.import_module(package).__name__ == package
