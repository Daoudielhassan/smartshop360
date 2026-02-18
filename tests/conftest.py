"""
tests/conftest.py
===================
Configuration pytest partagée entre tous les modules de test.
"""

import os
import sys
import pytest
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  Ajoute la racine du projet au sys.path
# ─────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────
#  Fixtures globales
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_df():
    """DataFrame générique réutilisable."""
    return pd.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "name":   ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "value":  [10.0, 20.0, 30.0, 40.0, 50.0],
        "active": [True, True, False, True, False],
    })


@pytest.fixture(scope="session")
def db_available():
    """Retourne True si la DB est accessible (tests d'intégration)."""
    try:
        # Import from the src directory which is added to sys.path
        from src.db.connection import get_connection
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False
