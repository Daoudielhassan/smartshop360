"""
src/agent/memory.py
=====================
Mémoire persistante des conversations — stocke l'historique dans PostgreSQL.

Table : conversations
  session_id  VARCHAR(100)
  role        VARCHAR(20)   -- 'user' | 'assistant'
  content     TEXT
  created_at  TIMESTAMP

Usage :
    from src.agent.memory import ConversationMemory
    mem = ConversationMemory(session_id="user_abc")
    mem.add("user", "Quel est le meilleur produit ?")
    mem.add("assistant", "D'après les données, c'est...")
    history = mem.get_history(last_n=10)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import text
from src.db_config import get_engine

_DDL_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(100) NOT NULL,
    role        VARCHAR(20)  NOT NULL,
    content     TEXT         NOT NULL,
    created_at  TIMESTAMP    DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, created_at DESC);
"""


def _ensure_table():
    """Crée la table conversations si elle n'existe pas encore."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            for stmt in _DDL_CONVERSATIONS.strip().split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(text(s))
    except Exception as e:
        print(f"[memory] Avertissement : impossible de créer la table conversations — {e}")


class ConversationMemory:
    """Gère la mémoire persistante d'une session de conversation."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        _ensure_table()

    def add(self, role: str, content: str) -> None:
        """Ajoute un message à l'historique PostgreSQL."""
        try:
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(text(
                    'INSERT INTO conversations (session_id, role, content) VALUES (:sid, :role, :content)'
                ), {"sid": self.session_id, "role": role, "content": content})
        except Exception as e:
            print(f"[memory] Erreur persistance message : {e}")

    def get_history(self, last_n: int = 20) -> list[dict]:
        """
        Retourne les N derniers messages de la session, du plus ancien au plus récent.
        Format : [{"role": "user"|"assistant", "content": "..."}]
        """
        try:
            engine = get_engine()
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT role, content FROM (
                        SELECT role, content, created_at
                        FROM conversations
                        WHERE session_id = :sid
                        ORDER BY created_at DESC
                        LIMIT :n
                    ) sub
                    ORDER BY created_at ASC
                """), {"sid": self.session_id, "n": last_n}).fetchall()
            return [{"role": r.role, "content": r.content} for r in rows]
        except Exception:
            return []

    def clear(self) -> None:
        """Supprime tout l'historique de la session."""
        try:
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(text(
                    "DELETE FROM conversations WHERE session_id = :sid"
                ), {"sid": self.session_id})
        except Exception as e:
            print(f"[memory] Erreur suppression historique : {e}")

    @staticmethod
    def list_sessions(limit: int = 50) -> list[dict]:
        """Liste les sessions récentes avec leur nb de messages."""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT session_id,
                           COUNT(*)           AS nb_messages,
                           MIN(created_at)    AS debut,
                           MAX(created_at)    AS dernier_msg
                    FROM conversations
                    GROUP BY session_id
                    ORDER BY dernier_msg DESC
                    LIMIT :limit
                """), {"limit": limit}).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception:
            return []
