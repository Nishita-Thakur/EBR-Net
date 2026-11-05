"""
conftest.py — makes the top-level packages (models, data, losses,
training, evaluation, inference) importable from tests/, regardless of
whether pytest is invoked as `pytest tests/` or `python -m pytest tests/`,
and regardless of the current working directory.

Without this, `tests/` (which has no __init__.py, by design) becomes
pytest's import root for each test file, so `from models... import ...`
fails with ModuleNotFoundError: No module named 'models' — the project
root never gets added to sys.path. Just having conftest.py exist here
is enough: pytest always adds this file's directory to sys.path when it
loads the root conftest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
