"""
SmartShop 360 - Application Streamlit
======================================
UI avec 4 écrans :
1. Dashboard KPIs unifiés (Ventes + Avis)
2. Analyse produit 360°
3. Assistant IA (Chatbot Data)
4. Qualité des Données (MDM Coverage)
"""

import math
import streamlit as st
import sqlite3
import pandas as pd
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()  # charge automatiquement .env si présent

# Chemin vers les modules
sys.path.insert(0, os.path.dirname(__file__))

from etl import run_etl, DB_PATH
from agent import run_agent, execute_sql, get_active_provider

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartShop 360",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2a5298;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .status-ok { color: #27ae60; font-weight: bold; }
    .status-warn { color: #f39c12; font-weight: bold; }
    .status-critical { color: #e74c3c; font-weight: bold; }
    .sql-box {
        background: #1e1e1e;
        color: #9cdcfe;
        padding: 10px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 12px;
        white-space: pre-wrap;
    }
    .chat-user {
        background: #e8f4fd;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 5px 0;
        border-left: 3px solid #2a5298;
    }
    .chat-bot {
        background: #f0f9f0;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 5px 0;
        border-left: 3px solid #27ae60;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INITIALISATION BASE DE DONNÉES
# ─────────────────────────────────────────────
@st.cache_resource
def init_db():
    """Lance l'ETL si la base n'existe pas."""
    if not os.path.exists(DB_PATH):
        with st.spinner("🔄 Initialisation de la base de données..."):
            run_etl()
    return DB_PATH

@st.cache_data(ttl=300)
def query_db(sql: str) -> pd.DataFrame:
    """Exécute une requête et retourne un DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        df = pd.DataFrame()
        st.error(f"Erreur SQL: {e}")
    finally:
        conn.close()
    return df

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shop.png", width=60)
        st.title("SmartShop 360")
        st.caption("Plateforme IA Intégrée")
        
        st.divider()
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Analyse Produit", "🤖 Assistant IA", "🏗️ Qualité des Données"],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.subheader("⚙️ Configuration LLM")
        api_key = st.text_input(
            "Clé API (Groq / Mistral / OpenAI / Anthropic)",
            type="password",
            help="Groq: gsk_... | Mistral: clé 32 chars | OpenAI: sk-... | Anthropic: sk-ant-...",
            placeholder="gsk_... / sk-... / sk-ant-..."
        )
        if api_key:
            # Stocker dans les variables d'env selon le préfixe
            if api_key.startswith("gsk_"):
                os.environ["GROQ_API_KEY"] = api_key
            elif api_key.startswith("sk-ant-"):
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif api_key.startswith("sk-"):
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                os.environ["MISTRAL_API_KEY"] = api_key

        provider_label = get_active_provider(api_key if api_key else "")
        st.caption(f"Provider actif : {provider_label}")
        
        st.divider()
        if st.button("🔄 Recharger les données"):
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        st.caption("SmartShop 360 — Projet E-commerce & IA")
    
    return page, api_key

# ─────────────────────────────────────────────
# PAGE 0 : DATA QUALITY (MDM Coverage)
# ─────────────────────────────────────────────
def render_data_quality():
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Qualité des Données & MDM</h1>
        <p>Taux de couverture du mapping produit — Référentiel commun (Golden Record)</p>
    </div>
    """, unsafe_allow_html=True)

    dq = query_db("SELECT * FROM V_DATA_QUALITY")
    if dq.empty:
        st.error("Vue V_DATA_QUALITY introuvable. Relancez l'ETL.")
        return

    row = dq.iloc[0]
    taux = float(row.get("Taux_Couverture_MDM", 0))

    # ── KPIs qualité ──────────────────────────
    st.subheader("📐 Indicateurs MDM")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🗂️ Produits ERP (Source 1)",     int(row["Nb_Produits_ERP"]))
    with c2:
        st.metric("💬 Produits avec Avis (Source 2)", int(row["Nb_Produits_Avis"]))
    with c3:
        st.metric("🔗 Mappings Golden Record",         int(row["Nb_Mappings"]))
    with c4:
        color = "🟢" if taux >= 90 else ("🟡" if taux >= 60 else "🔴")
        st.metric(f"{color} Taux de Couverture MDM", f"{taux:.1f} %")

    st.divider()

    # ── Jauge de couverture ───────────────────
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("📊 Couverture MDM")
        filled = math.floor(taux / 10)
        bar = "🟦" * filled + "⬜" * (10 - filled)
        st.markdown(f"""
        <div style='font-size:28px; letter-spacing:2px'>{bar}</div>
        <div style='font-size:16px; color:gray'>{taux:.1f} % des produits ERP ont au moins un avis client</div>
        """, unsafe_allow_html=True)

        st.divider()
        c5, c6 = st.columns(2)
        with c5:
            st.metric("📄 Avis Total",         int(row["Nb_Avis_Total"]))
            st.metric("🔗 Avis Associés",      int(row["Nb_Avis_Lies"]))
        with c6:
            st.metric("🧾 Factures",   int(row["Nb_Factures"]))
            st.metric("👤 Clients Uniques", int(row["Nb_Clients"]))

    with col_right:
        st.subheader("📋 Table de Mapping MDM (extrait)")
        mapping_df = query_db("""
        SELECT MappingID, ERP_ProductCode, ERP_ProductName, Category, GoldenRecordName
        FROM PRODUCT_MAPPING
        ORDER BY MappingID
        LIMIT 20
        """)
        if not mapping_df.empty:
            st.dataframe(mapping_df, use_container_width=True, height=350)

    st.divider()

    # ── Couverture par catégorie ───────────────
    st.subheader("📊 Couverture des Avis par Catégorie")
    cat_cov = query_db("""
    SELECT
        p.Category,
        COUNT(p.ProductID)                                    AS Nb_Produits,
        COUNT(DISTINCT r.ProduitID)                           AS Nb_Avec_Avis,
        ROUND(100.0 * COUNT(DISTINCT r.ProduitID)
              / COUNT(p.ProductID), 1)                        AS Couverture_Pct
    FROM PRODUCTS p
    LEFT JOIN REVIEWS r ON r.ProduitID = p.ProductID
    GROUP BY p.Category
    ORDER BY Couverture_Pct DESC
    """)
    if not cat_cov.empty:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.dataframe(cat_cov, use_container_width=True)
        with col_b:
            st.bar_chart(cat_cov.set_index("Category")["Couverture_Pct"])

    # ── Stratégie MDM ─────────────────────────
    st.divider()
    st.subheader("📖 Stratégie MDM — Approche du Projet")
    st.markdown("""
    | # | Approche | Utilisée dans ce projet |
    |---|----------|-------------------------|
    | 1 | **Codes communs EAN/GTIN/SKU** | ✅ Le `StockCode` ERP sert de Golden Key |
    | 2 | **Fuzzy Matching** (Levenshtein, Jaro-Winkler) | 🔲 Applicable si sources hétérogènes |
    | 3 | **Embedding Sémantique** (sentence-transformers) | 🔲 Recommandé en contexte multi-marketplace |
    | 4 | **MDM Commercial** (Informatica, Talend) | 🔲 Adapté aux SI enterprise |

    > **Approche retenue** : Les transactions Online Retail II et les avis sont unifiés via le `StockCode` 
    > (clé naturelle de l'ERP). La table `PRODUCT_MAPPING` joue le rôle de référentiel produit — 
    > permettant toutes les jointures SQL croisées entre ventes et réputation.
    """)

# ─────────────────────────────────────────────
# PAGE 1 : DASHBOARD
# ─────────────────────────────────────────────
def render_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Tableau de Bord SmartShop 360</h1>
        <p>Vue unifiée : Ventes × Satisfaction Client</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs globaux
    kpi_sql = """
    SELECT
        ROUND(SUM(CA), 0) as CA_Total,
        ROUND(SUM(Marge), 0) as Marge_Total,
        SUM(QuantiteVendue) as Quantites_Total,
        ROUND(AVG(Notemoyenne), 2) as Note_Globale,
        SUM(NbAvis) as Total_Avis,
        COUNT(*) as Nb_Produits
    FROM V_PRODUCT_KPI
    """
    kpis = query_db(kpi_sql)
    
    if not kpis.empty:
        row = kpis.iloc[0]
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("💰 CA Total", f"{int(row.CA_Total):,} €")
        with col2:
            st.metric("📈 Marge Totale", f"{int(row.Marge_Total):,} €")
        with col3:
            st.metric("📦 Unités Vendues", f"{int(row.Quantites_Total):,}")
        with col4:
            st.metric("⭐ Note Globale", f"{row.Note_Globale}/5")
        with col5:
            st.metric("💬 Avis Clients", f"{int(row.Total_Avis):,}")
        with col6:
            st.metric("🏷️ Produits", int(row.Nb_Produits))
    
    st.divider()
    
    # Tableau principal
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📋 Performances Produits")
        
        # Filtres
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            categories = ["Toutes"] + query_db("SELECT DISTINCT Category FROM PRODUCTS ORDER BY Category")["Category"].tolist()
            cat_filter = st.selectbox("Catégorie", categories)
        with fcol2:
            statut_filter = st.selectbox("Statut", ["Tous", "OK", "A_SURVEILLER", "CRITIQUE"])
        with fcol3:
            sort_by = st.selectbox("Trier par", ["CA ↓", "Note ↓", "Quantité ↓", "Avis Négatifs ↑"])
        
        sort_map = {
            "CA ↓": "CA DESC",
            "Note ↓": "Notemoyenne DESC",
            "Quantité ↓": "QuantiteVendue DESC",
            "Avis Négatifs ↑": "AvisNegatifs DESC"
        }

        # Valeurs autorisées pour éviter toute injection SQL
        VALID_STATUTS = {"OK", "A_SURVEILLER", "CRITIQUE"}

        where_clauses = []
        if cat_filter != "Toutes":
            # paramètre passé via ? pour éviter l'injection SQL
            where_clauses.append(f"Category = ?")
        if statut_filter != "Tous" and statut_filter in VALID_STATUTS:
            where_clauses.append(f"ProductID IN (SELECT ProductID FROM V_ALERTS WHERE Statut = ?)")
        
        where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        products_sql = f"""
        SELECT 
            p.ProductName as Produit,
            p.Category as Catégorie,
            p.CA as 'CA (€)',
            p.QuantiteVendue as 'Qté Vendue',
            ROUND(p.Notemoyenne, 1) as 'Note /5',
            p.NbAvis as 'Nb Avis',
            p.AvisPositifs as '👍',
            p.AvisNegatifs as '👎',
            COALESCE(a.Statut, 'OK') as Statut
        FROM V_PRODUCT_KPI p
        LEFT JOIN V_ALERTS a ON a.ProductID = p.ProductID
        {where}
        ORDER BY {sort_map[sort_by]}
        LIMIT 25
        """

        # Paramètres liés dans l'ordre des where_clauses
        params = []
        if cat_filter != "Toutes":
            params.append(cat_filter)
        if statut_filter != "Tous" and statut_filter in VALID_STATUTS:
            params.append(statut_filter)

        # Remplacement manuel des ? par les valeurs échappées (sqlite3 via query_db)
        safe_sql = products_sql
        for p in params:
            safe_sql = safe_sql.replace("?", f"'{p}'", 1)

        df_products = query_db(safe_sql)
        
        if not df_products.empty:
            def color_statut(val):
                colors = {"OK": "#27ae60", "A_SURVEILLER": "#f39c12", "CRITIQUE": "#e74c3c"}
                color = colors.get(val, "#333")
                return f"color: {color}; font-weight: bold"
            
            styled = df_products.style.map(color_statut, subset=["Statut"])
            st.dataframe(styled, use_container_width=True, height=400)
    
    with col_right:
        st.subheader("🚨 Alertes Actives")
        alerts_sql = """
        SELECT ProductName, Notemoyenne, AvisNegatifs, Statut
        FROM V_ALERTS
        WHERE Statut != 'OK'
        ORDER BY Notemoyenne ASC
        LIMIT 8
        """
        alerts = query_db(alerts_sql)
        
        if not alerts.empty:
            for _, row in alerts.iterrows():
                color = "🔴" if row["Statut"] == "CRITIQUE" else "🟡"
                st.markdown(f"""
                <div style="background:{'#fde8e8' if row['Statut'] == 'CRITIQUE' else '#fef9e7'};
                            padding:10px; border-radius:8px; margin:5px 0;
                            border-left: 3px solid {'#e74c3c' if row['Statut'] == 'CRITIQUE' else '#f39c12'}">
                    {color} <b>{row['ProductName'][:30]}</b><br>
                    ⭐ {row['Notemoyenne']}/5 | 👎 {int(row['AvisNegatifs'])} avis négatifs
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Aucune alerte active !")
        
        st.subheader("📊 Par Catégorie")
        cat_sql = """
        SELECT Category, ROUND(SUM(CA),0) as CA, ROUND(AVG(Notemoyenne),1) as Note
        FROM V_PRODUCT_KPI
        GROUP BY Category
        ORDER BY CA DESC
        """
        cat_df = query_db(cat_sql)
        if not cat_df.empty:
            st.bar_chart(cat_df.set_index("Category")["CA"])

# ─────────────────────────────────────────────
# PAGE 2 : ANALYSE PRODUIT 360°
# ─────────────────────────────────────────────
def render_product_analysis():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Analyse Produit 360°</h1>
        <p>Ventes + Avis clients combinés</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sélection produit
    products = query_db("SELECT ProductID, ProductName, Category FROM PRODUCTS ORDER BY ProductName")
    if products.empty:
        st.error("Aucun produit trouvé. Vérifiez que la base est initialisée.")
        return
    
    selected = st.selectbox(
        "Sélectionner un produit",
        products["ProductName"].tolist(),
        index=0
    )
    
    product_row = products[products["ProductName"] == selected].iloc[0]
    pid = product_row["ProductID"]
    
    # KPIs produit
    kpi = query_db(f"""
    SELECT CA, Marge, QuantiteVendue, Notemoyenne, NbAvis, AvisPositifs, AvisNegatifs, AvisNeutres
    FROM V_PRODUCT_KPI WHERE ProductID = '{pid}'
    """)
    
    if not kpi.empty:
        row = kpi.iloc[0]
        st.subheader(f"📦 {selected} — {product_row['Category']}")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("💰 CA", f"{int(row.CA):,} €")
        with c2:
            st.metric("📈 Marge", f"{int(row.Marge):,} €")
        with c3:
            st.metric("📦 Qté Vendue", int(row.QuantiteVendue))
        with c4:
            note_color = "🟢" if row.Notemoyenne >= 3.5 else ("🟡" if row.Notemoyenne >= 2.5 else "🔴")
            st.metric(f"{note_color} Note Moy.", f"{row.Notemoyenne}/5")
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Ventes dans le temps")
            sales_time = query_db(f"""
            SELECT 
                strftime('%Y-%m', i.Date) as Mois,
                ROUND(SUM(il.Revenue), 0) as CA
            FROM INVOICE_LINES il
            JOIN INVOICES i ON i.FactureID = il.FactureID
            WHERE il.ProduitID = '{pid}'
            GROUP BY Mois
            ORDER BY Mois
            """)
            if not sales_time.empty:
                st.line_chart(sales_time.set_index("Mois")["CA"])
            else:
                st.info("Pas de données de ventes temporelles")
        
        with col2:
            st.subheader("💬 Répartition des avis")
            if int(row.NbAvis) > 0:
                avis_data = pd.DataFrame({
                    "Type": ["Positifs 👍", "Négatifs 👎", "Neutres 😐"],
                    "Nombre": [int(row.AvisPositifs), int(row.AvisNegatifs), int(row.AvisNeutres)]
                })
                st.bar_chart(avis_data.set_index("Type"))
        
        # Derniers avis
        st.subheader("📝 Derniers avis clients")
        reviews = query_db(f"""
        SELECT ReviewText, Sentiment, Note, ReviewDate
        FROM REVIEWS
        WHERE ProduitID = '{pid}'
        ORDER BY ReviewDate DESC
        LIMIT 6
        """)
        
        if not reviews.empty:
            for _, rev in reviews.iterrows():
                emoji = "👍" if rev["Sentiment"] == "positive" else ("👎" if rev["Sentiment"] == "negative" else "😐")
                color = "#e8f8e8" if rev["Sentiment"] == "positive" else ("#fde8e8" if rev["Sentiment"] == "negative" else "#fef9e7")
                st.markdown(f"""
                <div style="background:{color}; padding:10px; border-radius:8px; margin:5px 0;">
                    {emoji} <b>{rev['Note']}/5</b> — {rev['ReviewText']}<br>
                    <small style="color:gray">{rev['ReviewDate'][:10]}</small>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3 : ASSISTANT IA
# ─────────────────────────────────────────────
def render_chat(api_key: str):
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Assistant IA SmartShop 360</h1>
        <p>Posez vos questions métier en langage naturel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation historique
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sql" not in st.session_state:
        st.session_state.show_sql = True
    
    # Options
    col1, col2 = st.columns([3, 1])
    with col2:
        st.session_state.show_sql = st.toggle("Afficher le SQL", value=True)
    
    # Questions d'exemple
    st.subheader("💡 Questions d'exemple")
    example_cols = st.columns(3)
    examples = [
        "Quels produits vendus à plus de 50 unités ont une note < 3 ?",
        "Quels sont nos 5 best-sellers en CA cette année ?",
        "Quels segments de clients sont les plus rentables ?",
        "Quelles catégories ont le meilleur sentiment client ?",
        "Quels produits ont beaucoup de ventes mais des avis négatifs ?",
        "Quel est le top 5 des clients par chiffre d'affaires ?",
    ]
    for i, ex in enumerate(examples):
        with example_cols[i % 3]:
            if st.button(f"💬 {ex[:40]}...", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = ex
    
    st.divider()
    
    # Historique de conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    👤 <b>Vous</b><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bot">
                    🤖 <b>Assistant SmartShop 360</b><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.show_sql and "sql" in msg:
                    with st.expander("🔍 Voir la requête SQL générée", expanded=False):
                        st.code(msg["sql"], language="sql")
                        if "data" in msg and msg["data"]:
                            df = pd.DataFrame(msg["data"])
                            st.dataframe(df, use_container_width=True)
    
    # Zone de saisie
    st.divider()
    
    # Traitement d'une question en attente (depuis les boutons d'exemple)
    pending = st.session_state.pop("pending_question", None)
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Votre question",
            value=pending or "",
            placeholder="Ex: Quels produits ont beaucoup de ventes mais de mauvais avis ?",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Envoyer 🚀", use_container_width=True)
    
    if submitted and user_input.strip():
        question = user_input.strip()
        
        # Ajoute la question à l'historique
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.spinner("🔍 Analyse en cours..."):
            result = run_agent(question, api_key if api_key else "")
        
        # Ajoute la réponse
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sql": result["sql"],
            "data": result["data"]
        })
        
        st.rerun()
    
    # Bouton reset
    if st.session_state.messages:
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Initialisation DB
    init_db()
    
    # Sidebar + navigation
    page, api_key = render_sidebar()
    
    # Rendu de la page sélectionnée
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔍 Analyse Produit":
        render_product_analysis()
    elif page == "🤖 Assistant IA":
        render_chat(api_key)
    elif page == "🏗️ Qualité des Données":
        render_data_quality()

if __name__ == "__main__":
    main()
