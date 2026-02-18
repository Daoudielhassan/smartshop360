"""
src/agent/rag.py
==================
Recherche sémantique sur les avis clients via pgvector.

Prérequis PostgreSQL :
    CREATE EXTENSION IF NOT EXISTS vector;

Installation :
    pip install pgvector sentence-transformers

Les avis sont vectorisés avec sentence-transformers (all-MiniLM-L6-v2)
et stockés dans la table review_embeddings.

Usage :
    from src.agent.rag import ReviewRAG
    rag = ReviewRAG()
    rag.build_index()                          # une seule fois
    results = rag.search("problème emballage", k=5)
"""

from __future__ import annotations
import os
import numpy as np
from sqlalchemy import text
from src.db_config import get_engine

_DDL_EMBEDDINGS = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS review_embeddings (
    review_id   INTEGER PRIMARY KEY REFERENCES review_facts("ReviewID") ON DELETE CASCADE,
    embedding   vector(384)
);
CREATE INDEX IF NOT EXISTS idx_review_embed ON review_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
"""

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model():
    """Charge le modèle d'embedding (lazy import)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(_MODEL_NAME)
    except ImportError:
        raise ImportError(
            "sentence-transformers requis pour le RAG.\n"
            "Installez avec : pip install sentence-transformers"
        )


class ReviewRAG:
    """Index de recherche sémantique sur les avis clients."""

    def __init__(self):
        self.engine = get_engine()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def setup_pgvector(self) -> None:
        """Active l'extension pgvector et crée la table des embeddings."""
        with self.engine.begin() as conn:
            for stmt in _DDL_EMBEDDINGS.strip().split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(text(s))
        print("[rag] Extension pgvector activée + table review_embeddings créée")

    def build_index(self, batch_size: int = 256) -> None:
        """
        Vectorise tous les avis non encore indexés et les stocke dans PostgreSQL.
        Peut être relancé de façon incrémentale — ignore les avis déjà indexés.
        """
        self.setup_pgvector()

        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT rf."ReviewID", rf."ReviewText"
                FROM review_facts rf
                LEFT JOIN review_embeddings re ON re.review_id = rf."ReviewID"
                WHERE re.review_id IS NULL
            """)).fetchall()

        if not rows:
            print("[rag] Aucun nouvel avis à indexer — index déjà à jour")
            return

        print(f"[rag] Vectorisation de {len(rows)} avis ...")
        texts = [r.ReviewText or "" for r in rows]
        ids   = [r.ReviewID for r in rows]

        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

        with self.engine.begin() as conn:
            for review_id, emb in zip(ids, embeddings):
                conn.execute(text(
                    "INSERT INTO review_embeddings (review_id, embedding) VALUES (:id, :emb) "
                    "ON CONFLICT (review_id) DO UPDATE SET embedding = EXCLUDED.embedding"
                ), {"id": review_id, "emb": emb.tolist()})

        print(f"[rag] {len(rows)} embeddings stockés dans PostgreSQL")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Recherche les k avis les plus proches sémantiquement de la requête.

        Returns
        -------
        Liste de dicts : {review_id, review_text, sentiment, rating, product_name, similarity}
        """
        query_emb = self.model.encode([query])[0].tolist()

        with self.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    rf."ReviewID"    AS review_id,
                    rf."ReviewText"  AS review_text,
                    rf."Sentiment"   AS sentiment,
                    rf."Rating"      AS rating,
                    COALESCE(pm."GoldenRecordName", rf."ProductID") AS product_name,
                    1 - (re.embedding <=> :emb::vector) AS similarity
                FROM review_embeddings re
                JOIN review_facts rf ON rf."ReviewID" = re.review_id
                LEFT JOIN product_mapping pm ON pm."Review_ProductCode" = rf."ProductID"
                ORDER BY re.embedding <=> :emb::vector
                LIMIT :k
            """), {"emb": str(query_emb), "k": k}).fetchall()

        return [dict(r._mapping) for r in rows]

    def search_formatted(self, query: str, k: int = 5) -> str:
        """Retourne les résultats de recherche formatés pour l'agent IA."""
        results = self.search(query, k=k)
        if not results:
            return "Aucun avis trouvé pour cette recherche sémantique."

        lines = [f" Résultats RAG pour : '{query}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r['product_name']}] Note:{r['rating']:.1f} "
                f"({r['sentiment']}) — similarité {r['similarity']:.2f}\n"
                f"   {r['review_text'][:200]}..."
            )
        return "\n".join(lines)
