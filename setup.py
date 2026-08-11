"""
Minimal setuptools shim.

All real package metadata lives in ``pyproject.toml`` (PEP 621) -- this
file doesn't duplicate any of it. It exists only so that
``pip install -e .`` also works on older pip/setuptools versions that
don't yet support PEP 660 (pyproject.toml-only editable installs), which
is common on shared/HPC Python installations that can lag well behind
the latest pip release. Without this file, those older pip versions fail
with something like::

    ERROR: File "setup.py" not found. Directory cannot be installed in
    editable mode: ...
    (A "pyproject.toml" file was found, but editable mode currently
    requires a setup.py based build.)

With it present, pip falls back to the classic setup.py-based editable
install path instead, which every supported pip version understands.

If ``pip install -e .`` still doesn't work in your environment (e.g. no
network access to fetch build dependencies), you don't need to install
the package at all -- PyChap works just as well by adding the repository
root to ``PYTHONPATH`` and running ``python -m pychap.cli ...`` directly;
see README.md.
"""

from setuptools import setup

setup()
