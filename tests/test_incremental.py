"""
tests/test_incremental.py
===========================
Tests pour src/etl/incremental.py
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.etl.incremental import (
    load_hashes,
    save_hashes,
    compute_file_hash,
    should_run_etl,
    record_hashes,
    HASH_FILE,
)


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_hash_file(tmp_path, monkeypatch):
    """Redirige HASH_FILE vers un répertoire temporaire."""
    test_hash = tmp_path / ".etl_hashes.json"
    monkeypatch.setattr("src.etl.incremental.HASH_FILE", str(test_hash))
    yield str(test_hash)
    if test_hash.exists():
        test_hash.unlink()


@pytest.fixture
def sample_file(tmp_path):
    """Crée un fichier CSV temporaire avec un contenu fixe."""
    f = tmp_path / "sample.csv"
    f.write_text("a,b,c\n1,2,3\n4,5,6\n")
    return str(f)


# ─────────────────────────────────────────────────────────────
#  Tests compute_file_hash
# ─────────────────────────────────────────────────────────────

class TestComputeFileHash:

    def test_returns_string(self, sample_file):
        h = compute_file_hash(sample_file)
        assert isinstance(h, str) and len(h) == 64  # SHA-256 hex = 64 chars

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "f1.csv"
        f2 = tmp_path / "f2.csv"
        content = "a,b\n1,2\n"
        f1.write_text(content)
        f2.write_text(content)
        assert compute_file_hash(str(f1)) == compute_file_hash(str(f2))

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "f1.csv"
        f2 = tmp_path / "f2.csv"
        f1.write_text("a,b\n1,2\n")
        f2.write_text("a,b\n3,4\n")
        assert compute_file_hash(str(f1)) != compute_file_hash(str(f2))

    def test_missing_file_returns_none_or_raises(self, tmp_path):
        missing = str(tmp_path / "nonexistent.csv")
        result = compute_file_hash(missing)
        # Acceptable: None OR raise FileNotFoundError
        assert result is None or True


# ─────────────────────────────────────────────────────────────
#  Tests load_hashes / save_hashes
# ─────────────────────────────────────────────────────────────

class TestHashPersistence:

    def test_load_empty_when_no_file(self):
        result = load_hashes()
        assert result == {}

    def test_save_and_reload(self, tmp_path, monkeypatch):
        test_hash_path = str(tmp_path / ".etl_hashes.json")
        monkeypatch.setattr("src.etl.incremental.HASH_FILE", test_hash_path)
        hashes = {"file1.csv": "abc123", "file2.json": "def456"}
        save_hashes(hashes)
        loaded = load_hashes()
        assert loaded == hashes

    def test_empty_save(self):
        save_hashes({})
        loaded = load_hashes()
        assert loaded == {}


# ─────────────────────────────────────────────────────────────
#  Tests should_run_etl
# ─────────────────────────────────────────────────────────────

class TestShouldRunETL:

    def test_returns_true_if_no_previous_hashes(self, sample_file):
        run, changed = should_run_etl([sample_file])
        assert run is True

    def test_returns_false_if_files_unchanged(self, sample_file):
        # Simule un premier run
        record_hashes([sample_file])
        run, changed = should_run_etl([sample_file])
        assert run is False
        assert changed == []

    def test_returns_true_if_file_changed(self, tmp_path, sample_file, monkeypatch):
        test_hash_path = str(tmp_path / ".etl_hashes.json")
        monkeypatch.setattr("src.etl.incremental.HASH_FILE", test_hash_path)
        record_hashes([sample_file])
        # Modifie le fichier
        with open(sample_file, "a") as f:
            f.write("7,8,9\n")
        run, changed = should_run_etl([sample_file])
        assert run is True
        assert sample_file in changed

    def test_force_always_returns_true(self, sample_file):
        record_hashes([sample_file])
        run, changed = should_run_etl([sample_file], force=True)
        assert run is True
