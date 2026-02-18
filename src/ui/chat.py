"""
src/ui/chat.py
===============
Interface de chat Streamlit pour l'assistant IA (Text-to-SQL).
Connecté à src/agent/graph.py.
"""

import streamlit as st
import pandas as pd


EXAMPLE_QUESTIONS = [
    "Quels sont les 5 produits avec le meilleur CA ?",
    "Quels produits ont une note inférieure à 3/5 ?",
    "Montre-moi les alertes critiques.",
    "Quel est le CA total par catégorie ?",
    "Quels clients ont le plus de commandes ?",
    "Quels produits sont vendus à plus de 100 unités avec une mauvaise note ?",
]


def render_chat(api_key: str, run_agent_fn):
    """
    Écran chatbot.

    Parameters
    ----------
    api_key     : clé API LLM (peut être None → mode fallback)
    run_agent_fn: fonction run_agent(question, api_key, history) → dict
    """
    st.title(" Assistant IA — SmartShop 360")
    st.caption("Posez vos questions en langage naturel — l'agent génère et exécute le SQL pour vous.")

    # Init historique dans session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_history" not in st.session_state:
        st.session_state.llm_history  = []

    # ── Questions exemples ────────────────────────────────────
    with st.expander(" Exemples de questions"):
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"ex_{i}"):
                    st.session_state["pending_question"] = q

    # ── Affichage historique ──────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sql"):
                with st.expander(" Voir le SQL généré"):
                    st.code(msg["sql"], language="sql")
                if msg.get("data"):
                    df = pd.DataFrame(msg["data"])
                    st.dataframe(df, width='stretch', hide_index=True)

    # ── Zone de saisie ────────────────────────────────────────
    pending  = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Posez votre question ici...")
    question = pending or user_input

    if question:
        # Affiche message utilisateur
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Appel agent
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                result = run_agent_fn(
                    question,
                    api_key,
                    st.session_state.llm_history,
                )

            st.markdown(result["answer"])

            if result.get("sql"):
                with st.expander(" Voir le SQL généré"):
                    st.code(result["sql"], language="sql")

            if result.get("data"):
                df = pd.DataFrame(result["data"])
                st.dataframe(df, width='stretch', hide_index=True)
                st.caption(f"↳ {result['row_count']} ligne(s) retournée(s)")

        # Mise à jour de l'historique LLM (contexte pour le prochain tour)
        st.session_state.llm_history.append({"role": "user",      "content": question})
        st.session_state.llm_history.append({"role": "assistant", "content": result["answer"]})

        # Sauvegarde dans l'historique affiché
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": result["answer"],
            "sql":     result.get("sql"),
            "data":    result.get("data", []),
        })

    # ── Bouton reset ──────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button(" Effacer la conversation"):
            st.session_state.chat_history = []
            st.session_state.llm_history  = []
            st.rerun()
