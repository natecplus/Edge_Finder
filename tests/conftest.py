"""Tests never touch your real database: point EDGE_DATA_DIR at a temp folder
before any edge module is imported."""
import os
import tempfile

os.environ["EDGE_DATA_DIR"] = tempfile.mkdtemp(prefix="edge_test_")
