"""
SmartShop 360 — Point d'entrée Streamlit
=========================================
Lance avec :  streamlit run app.py

Architecture suivie : architecture.txt
  - src/db_config.py          : connexion PostgreSQL (SQLAlchemy)
  - src/etl/run_etl.py        : pipeline ETL complet
  - src/agent/graph.py        : agent Text-to-SQL (multi-LLM)
  - src/ui/dashboard.py       : composants graphiques Plotly
  - src/ui/chat.py            : interface chatbot
  - src/ui/analytics.py       : pages analytiques avancées

10 écrans :
  1.  📊 Dashboard              — KPIs globaux + alertes
  2.  🔍 Analyse Produit        — Fiche 360° par produit
  3.  🤖 Assistant IA           — Chatbot Text-to-SQL
  4.  🏗️  Qualité des Données   — Couverture MDM
  5.  📅 Analyse Temporelle     — Filtres date + CA par période
  6.  🗺️  Carte Géographique    — Choropleth par pays
  7.  ⚖️  Comparaison Produits  — Radar + barres côte à côte
  8.  🏅 Scoring Produits       — Composite CA/Note/Qté/Avis
  9.  ⚠️  Churn Clients         — RFM + RandomForest
  10. 📈 Prévision Ventes       — Prophet / régression linéaire
"""

import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Imports modules src/ ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.db_config import get_engine, test_connection
from src.agent.graph import run_agent, get_active_provider
from src.ui.dashboard import render_dashboard, render_product_analysis, render_data_quality
from src.ui.chat import render_chat
from src.ui.analytics import (
    render_temporal_filters,
    render_geo_map,
    render_product_comparison,
    render_scoring,
    render_churn,
    render_forecast,
)

# Métriques Prometheus (optionnel)
try:
    from monitoring.prometheus_metrics import start_metrics_server
    start_metrics_server()
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration globale Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartShop 360",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Init BDD (ETL au premier lancement)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initialisation de la base de données...")
def init_db():
    """
    Vérifie la connexion PostgreSQL.
    Si les tables sont vides, lance l'ETL automatiquement.
    """
    if not test_connection():
        st.error(
            "❌ Impossible de se connecter à PostgreSQL.\n\n"
            "Lancez le conteneur Docker : `docker-compose up -d db`\n"
            "puis rechargez cette page."
        )
        st.stop()

    engine = get_engine()
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            count = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
    except Exception:
        count = 0

    if count == 0:
        from src.etl.run_etl import run_etl
        run_etl(force=True)   # force=True : ignore les avertissements qualité au 1er démarrage

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Requête BDD (cachée 5 min)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def query_db(sql: str) -> pd.DataFrame:
    """Exécute une requête SELECT et retourne un DataFrame."""
    try:
        engine = get_engine()
        return pd.read_sql(sql, engine)
    except Exception as e:
        st.warning(f"Erreur requête : {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    """Retourne (page_sélectionnée, api_key)."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/shopping-cart--v2.png",
            width=64,
        )
        st.title("SmartShop 360")
        st.caption("Data + IA — E-commerce B2C")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "📊 Dashboard",
                "🔍 Analyse Produit",
                "🤖 Assistant IA",
                "🏗️ Qualité des Données",
                "📅 Analyse Temporelle",
                "🗺️ Carte Géographique",
                "⚖️ Comparaison Produits",
                "🏅 Scoring Produits",
                "⚠️ Churn Clients",
                "📈 Prévision Ventes",
            ],
        )

        st.divider()
        st.subheader("🔑 Clé API LLM")

        env_key = (
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("MISTRAL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or ""
        )

        api_key = st.text_input(
            "Clé API (Groq / Mistral / OpenAI / Anthropic)",
            value=env_key,
            type="password",
            placeholder="gsk_... / sk-... / sk-ant-...",
            help="Laissez vide pour utiliser les variables d'environnement du .env",
        )

        if api_key:
            if api_key.startswith("gsk_"):
                os.environ["GROQ_API_KEY"] = api_key
            elif api_key.startswith("sk-ant-"):
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif api_key.startswith("sk-"):
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                os.environ["MISTRAL_API_KEY"] = api_key

        provider_label = get_active_provider(api_key or None)
        st.info(f"**Provider actif** : {provider_label}")

        st.divider()
        if st.button("🔄 Relancer l'ETL"):
            st.cache_data.clear()
            st.cache_resource.clear()
            from src.etl.run_etl import run_etl
            with st.spinner("ETL en cours..."):
                run_etl()
            st.success("ETL terminé !")
            st.rerun()

        st.caption("SmartShop 360 © 2026")

    return page, api_key


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_db()
    page, api_key = render_sidebar()

    if page == "📊 Dashboard":
        render_dashboard(query_db)

    elif page == "🔍 Analyse Produit":
        render_product_analysis(query_db)

    elif page == "🤖 Assistant IA":
        render_chat(api_key or "", run_agent)

    elif page == "🏗️ Qualité des Données":
        render_data_quality(query_db)

    elif page == "📅 Analyse Temporelle":
        render_temporal_filters(query_db)

    elif page == "🗺️ Carte Géographique":
        render_geo_map(query_db)

    elif page == "⚖️ Comparaison Produits":
        render_product_comparison(query_db)

    elif page == "🏅 Scoring Produits":
        render_scoring(query_db)

    elif page == "⚠️ Churn Clients":
        render_churn(query_db)

    elif page == "📈 Prévision Ventes":
        render_forecast(query_db)


if __name__ == "__main__":
    main()

