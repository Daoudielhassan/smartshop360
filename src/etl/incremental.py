"""
src/etl/incremental.py
========================
Moteur d'ETL incrémental — détecte si les fichiers source ont changé
via SHA-256 et évite un rechargement complet si rien n'a changé.

Usage :
    from src.etl.incremental import should_run_etl, record_hashes
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

HASH_FILE = Path(__file__).parent.parent.parent / ".etl_hashes.json"


def _file_sha256(path: str) -> str:
    """Calcule le SHA-256 d'un fichier (lecture par blocs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_file_hash(path: str) -> str | None:
    """Calcule le SHA-256 d'un fichier. Retourne None si le fichier est introuvable."""
    try:
        return _file_sha256(path)
    except FileNotFoundError:
        return None


def load_hashes() -> dict:
    """Charge les hashes précédemment enregistrés."""
    p = Path(HASH_FILE)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict) -> None:
    """Persiste les hashes dans le fichier JSON."""
    hashes["_last_run"] = datetime.utcnow().isoformat()
    p = Path(HASH_FILE)
    with open(p, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"[incremental] Hashes enregistrés → {p}")


def compute_current_hashes(file_paths: list[str]) -> dict:
    """Calcule les hashes actuels pour une liste de fichiers."""
    return {p: _file_sha256(p) for p in file_paths if os.path.exists(p)}


def should_run_etl(file_paths: list[str], force: bool = False) -> tuple[bool, list[str]]:
    """
    Détermine si l'ETL doit tourner.

    Returns
    -------
    (should_run: bool, changed_files: list[str])
        should_run     → True si au moins un fichier a changé ou si force=True
        changed_files  → liste des fichiers modifiés depuis le dernier run
    """
    if force:
        print("[incremental] Mode FORCE — ETL complet lancé")
        return True, file_paths

    previous = load_hashes()
    current  = compute_current_hashes(file_paths)
    changed  = [
        p for p, h in current.items()
        if previous.get(p) != h
    ]

    if changed:
        print(f"[incremental] {len(changed)} fichier(s) modifié(s) : {changed}")
        return True, changed
    else:
        last = previous.get("_last_run", "jamais")
        print(f"[incremental] Aucun changement détecté (dernier run : {last}) — ETL ignoré")
        return False, []


def record_hashes(file_paths: list[str]) -> None:
    """Enregistre les hashes après un ETL réussi."""
    current = compute_current_hashes(file_paths)
    save_hashes(current)
