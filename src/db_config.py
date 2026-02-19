"""
SmartShop 360 – Connexion PostgreSQL (SQLAlchemy)
==================================================
Lit les variables d'environnement définies dans .env
pour construire l'URL de connexion.

Usage :
    from src.db_config import get_engine, get_connection
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ─── Paramètres de connexion ────────────────────────────────────────────────
POSTGRES_USER     = os.environ.get("POSTGRES_USER",     "admin")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "password")
POSTGRES_DB       = os.environ.get("POSTGRES_DB",       "smartshop_db")
POSTGRES_HOST     = os.environ.get("POSTGRES_HOST",     "localhost")
POSTGRES_PORT     = os.environ.get("POSTGRES_PORT",     "5432")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


def get_engine():
    """Retourne un moteur SQLAlchemy (connexion poolée et robuste)."""
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,       # teste la connexion avant chaque usage
        pool_recycle=1800,        # recycle les connexions après 30 min (évite les connexions mortes)
        pool_size=5,
        max_overflow=10,
        connect_args={
            "connect_timeout": 10,   # abandonne si le serveur ne répond pas en 10 s
            "options": "-c statement_timeout=30000",  # coupe une requête bloquante après 30 s
        },
    )


def get_connection():
    """Retourne une connexion brute psycopg2 (usage ETL)."""
    import psycopg2
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def test_connection() -> bool:
    """Vérifie que la BDD est accessible."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[db_config] Connexion échouée : {e}")
        return False
