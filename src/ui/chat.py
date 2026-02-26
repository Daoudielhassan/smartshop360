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
    "Quels segments de clients sont rentables ET satisfaits ?",
    "Analyse les profils RFM de nos clients.",
    "Fais un clustering automatique de nos clients.",
]

# ── Couleurs par segment ──────────────────────────────────
_SEGMENT_COLORS = {
    "Champions":           "#2ecc71",
    "Déçus Rentables":     "#e67e22",
    "Fans Peu Dépensiers": "#3498db",
    "Inactifs":            "#95a5a6",
}



def _render_analysis(analysis: dict) -> None:
    """Affiche les résultats d'analyse Python (segmentation, RFM, etc.)."""
    if not analysis:
        return
    result = analysis.get("result", {})
    if not isinstance(result, dict):
        return

    atype = result.get("type")

    # ── Segmentation quadrants ────────────────────────────
    if atype == "segmentation":
        segments    = result.get("segments", {})
        seuil_ca    = result.get("seuil_CA", 0)
        seuil_note  = result.get("seuil_note", 0)
        total       = result.get("total", 0)

        st.markdown("#### Segmentation Clients — Rentabilité × Satisfaction")
        st.caption(
            f"Seuil CA médian : **{seuil_ca:,.0f} €** | "
            f"Seuil Note médian : **{seuil_note:.2f} / 5** | "
            f"**{total}** clients analysés"
        )

        cols = st.columns(4)
        for i, (seg_name, seg_data) in enumerate(segments.items()):
            color = _SEGMENT_COLORS.get(seg_name, "#ccc")
            with cols[i % 4]:
                st.markdown(
                    f"""<div style="border-left:4px solid {color};padding:8px 12px;
                    border-radius:4px;background:#f8f9fa;margin-bottom:8px">
                    <strong>{seg_name}</strong><br/>
                    <span style="font-size:1.4em;font-weight:700">{seg_data['count']}</span>
                    <span style="color:#666"> clients</span><br/>
                    CA moy : <strong>{seg_data['CA_moyen']:,.0f} €</strong><br/>
                    Note moy : <strong>{seg_data['Note_moyenne']:.2f} ★</strong>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if seg_data.get("top_clients"):
                    top_df = pd.DataFrame(seg_data["top_clients"])
                    st.dataframe(top_df, hide_index=True, use_container_width=True)

    # ── RFM ──────────────────────────────────────────────
    elif atype == "rfm":
        profils = result.get("profils", {})
        total   = result.get("total", 0)
        st.markdown("#### Scoring RFM — Profils Clients")
        st.caption(f"**{total}** clients scorés (Fréquence × Montant)")
        rfm_df = pd.DataFrame.from_dict(profils, orient="index").reset_index()
        rfm_df.columns = ["Profil", "Effectif", "CA moyen (€)"]
        st.dataframe(rfm_df, hide_index=True, use_container_width=True)
    # ── Clustering K-Means ───────────────────────────
    elif atype == "clustering":
        clusters   = result.get("clusters", {})
        k          = result.get("k", 0)
        total      = result.get("total", 0)
        importance = result.get("variable_importance", {})

        st.markdown(f"#### 🧠 Clustering K-Means — {k} groupes détectés")
        st.caption(f"**{total}** clients analysés — segmentation non supervisée")

        cols = st.columns(k)
        _CLUSTER_COLORS = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
        for i, (cluster_name, cdata) in enumerate(clusters.items()):
            color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            with cols[i % k]:
                st.markdown(
                    f"""<div style="border-left:4px solid {color};padding:8px 12px;
                    border-radius:4px;background:#f8f9fa;margin-bottom:8px">
                    <strong>{cluster_name}</strong><br/>
                    <span style="font-size:1.3em;font-weight:700">{cdata['count']}</span>
                    <span style="color:#666"> clients</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                moyennes = cdata.get("moyennes", {})
                if moyennes:
                    moy_df = pd.DataFrame(
                        [{"Variable": k2, "Moyenne": v} for k2, v in moyennes.items()
                         if k2 != "_Cluster"]
                    )
                    st.dataframe(moy_df, hide_index=True, use_container_width=True)

        if importance:
            with st.expander("📊 Variables les plus discriminantes"):
                imp_df = pd.DataFrame(
                    [{"Variable": k2, "Dispersion inter-cluster": v}
                     for k2, v in importance.items()]
                )
                st.dataframe(imp_df, hide_index=True, use_container_width=True)
    # ── Summary / Correlation (texte générique) ───────────
    elif isinstance(result, dict) and result:
        with st.expander("Analyse statistique complémentaire"):
            st.json(result)


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
                _render_analysis(msg.get("analysis"))

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

            _render_analysis(result.get("analysis"))

        # Mise à jour de l'historique LLM (contexte pour le prochain tour)
        st.session_state.llm_history.append({"role": "user",      "content": question})
        st.session_state.llm_history.append({"role": "assistant", "content": result["answer"]})

        # Sauvegarde dans l'historique affiché
        st.session_state.chat_history.append({
            "role":     "assistant",
            "content":  result["answer"],
            "sql":      result.get("sql"),
            "data":     result.get("data", []),
            "analysis": result.get("analysis"),
        })

    # ── Bouton reset ──────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button(" Effacer la conversation"):
            st.session_state.chat_history = []
            st.session_state.llm_history  = []
            st.rerun()
