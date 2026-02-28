"""
src/ui/dashboard.py
====================
NOTE: All SQL computed aliases use lowercase to avoid PostgreSQL case-folding issues.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _kpi(label: str, value, delta=None, help_text: str = ""):
    st.metric(label=label, value=value, delta=delta, help=help_text)


def _statut_color(statut: str) -> str:
    return {"CRITIQUE": "red", "A_SURVEILLER": "orange", "OK": "green"}.get(statut, "")


def render_dashboard(query_db):
    st.title("Dashboard SmartShop 360")

    kpi_data = query_db("SELECT * FROM v_data_quality")
    prod_kpi = query_db(
        'SELECT SUM("CA") AS ca_total, AVG("Notemoyenne") AS note_moy, '
        'SUM("NbAvis") AS nb_avis, COUNT(*) AS nb_produits FROM v_product_kpi'
    )

    cols = st.columns(4)
    if not prod_kpi.empty:
        row = prod_kpi.iloc[0]
        with cols[0]: _kpi("CA Total", f"{row['ca_total']:,.0f} EUR")
        with cols[1]: _kpi("Note Moyenne", f"{row['note_moy']:.1f} / 5")
        with cols[2]: _kpi("Total Avis", f"{int(row['nb_avis']):,}")
        with cols[3]: _kpi("Produits", f"{int(row['nb_produits'])}")

    if not kpi_data.empty:
        taux = float(kpi_data.iloc[0].get("Taux_Couverture_MDM", 0))
        st.info(f"Couverture MDM : {taux:.0f}%")

    st.divider()

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        cats = ["Toutes"] + query_db(
            'SELECT DISTINCT "Category" FROM products ORDER BY "Category"'
        )["Category"].tolist()
        cat_filter = st.selectbox("Categorie", cats)
    with fcol2:
        statut_filter = st.selectbox("Statut", ["Tous", "OK", "A_SURVEILLER", "CRITIQUE"])
    with fcol3:
        sort_by = st.selectbox("Trier par", ["CA", "Note", "Quantite", "Avis Negatifs"])

    sort_map = {
        "CA":            'p."CA" DESC',
        "Note":          'p."Notemoyenne" DESC',
        "Quantite":      'p."QuantiteVendue" DESC',
        "Avis Negatifs": 'p."AvisNegatifs" DESC',
    }
    VALID_STATUTS = {"OK", "A_SURVEILLER", "CRITIQUE"}

    where_parts = []
    if cat_filter != "Toutes":
        where_parts.append('p."Category" = \'{}\''.format(cat_filter.replace("'", "''")))
    if statut_filter != "Tous" and statut_filter in VALID_STATUTS:
        where_parts.append(
            'p."ProductID" IN (SELECT "ProductID" FROM v_alerts WHERE "Statut" = \'{}\')'.format(statut_filter)
        )

    where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
    products_sql = f"""
        SELECT
            p."ProductName"      AS produit,
            p."Category"         AS categorie,
            p."CA"               AS ca_eur,
            p."QuantiteVendue"   AS qte_vendue,
            ROUND(p."Notemoyenne"::NUMERIC, 1) AS note,
            p."NbAvis"           AS nb_avis,
            p."AvisPositifs"     AS avis_pos,
            p."AvisNegatifs"     AS avis_neg,
            COALESCE(a."Statut", 'OK') AS statut
        FROM v_product_kpi p
        LEFT JOIN v_alerts a ON a."ProductID" = p."ProductID"
        {where}
        ORDER BY {sort_map[sort_by]}
        LIMIT 25
    """
    df_products = query_db(products_sql)

    if not df_products.empty:
        df_display = df_products.rename(columns={
            "produit": "Produit", "categorie": "Categorie", "ca_eur": "CA (EUR)",
            "qte_vendue": "Qte Vendue", "note": "Note /5", "nb_avis": "Nb Avis",
            "avis_pos": "Positifs", "avis_neg": "Negatifs", "statut": "Statut",
        })

        def color_statut(val):
            m = {"CRITIQUE": "background-color:#ffcccc", "A_SURVEILLER": "background-color:#fff3cd"}
            return m.get(val, "")

        styled = df_display.style.map(color_statut, subset=["Statut"])
        st.dataframe(styled, width='stretch', hide_index=True)
    else:
        st.info("Aucun produit trouve avec ces filtres.")

    st.divider()

    cat_data = query_db(
        'SELECT "Category" AS category, SUM("CA") AS ca FROM v_product_kpi '
        'GROUP BY "Category" ORDER BY ca DESC'
    )
    if not cat_data.empty:
        fig = px.bar(
            cat_data, x="category", y="ca",
            color="ca", color_continuous_scale="Blues",
            title="Chiffre d Affaires par Categorie",
            labels={"category": "Categorie", "ca": "CA (EUR)"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, width='stretch')

    alerts = query_db(
        'SELECT "ProductName" AS pname, "Notemoyenne" AS note, '
        '"AvisNegatifs" AS aneg, "Statut" AS statut '
        'FROM v_alerts WHERE "Statut" != \'OK\' ORDER BY "Notemoyenne" ASC LIMIT 10'
    )
    if not alerts.empty:
        st.subheader("Alertes Produits")
        for _, row in alerts.iterrows():
            st.warning(
                f"**{row['pname']}**  Note : {row['note']:.1f}/5  {int(row['aneg'])} avis negatifs  Statut: {row['statut']}"
            )


def render_product_analysis(query_db):
    st.title("Analyse Produit 360")

    products = query_db('SELECT "ProductID" AS pid, "ProductName" AS pname FROM products ORDER BY "ProductName"')
    if products.empty:
        st.warning("Aucun produit disponible.")
        return

    product_map = dict(zip(products["pname"], products["pid"]))
    selected_name = st.selectbox("Choisissez un produit", list(product_map.keys()))
    selected_id   = product_map[selected_name]

    kpi = query_db(f"""
        SELECT
            ROUND(COALESCE(SUM(sf."Revenue"), 0)::NUMERIC, 2)  AS ca,
            COALESCE(SUM(sf."Quantity"), 0)                    AS qte,
            COUNT(DISTINCT sf."InvoiceNo")                     AS nb_factures,
            COUNT(*)                                           AS nb_lignes
        FROM sales_facts sf
        WHERE sf."StockCode" = '{selected_id}'
    """)

    # Récupère note et avis via product_mapping si disponible, sinon 0
    kpi_reviews = query_db(f"""
        SELECT
            ROUND(AVG(rf."Rating")::NUMERIC, 2)  AS note,
            COUNT(rf."ReviewID")                 AS nb_avis
        FROM product_mapping pm
        JOIN review_facts rf ON rf."ProductID" = pm."Review_ProductCode"
        WHERE pm."ERP_StockCode" = '{selected_id}'
    """)

    if kpi.empty or kpi.iloc[0]["nb_lignes"] == 0:
        st.warning("Aucune transaction trouvée pour ce produit.")
        return

    row = kpi.iloc[0]
    note    = kpi_reviews.iloc[0]["note"]    if not kpi_reviews.empty else None
    nb_avis = kpi_reviews.iloc[0]["nb_avis"] if not kpi_reviews.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: _kpi("CA",          f"{row['ca']:,.0f} EUR")
    with c2: _kpi("Note",        f"{note:.1f} / 5" if note else "N/A")
    with c3: _kpi("Qte Vendue",  f"{int(row['qte']):,}")
    with c4: _kpi("Avis",        f"{int(nb_avis):,}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Evolution des Ventes (mensuel)")
        sales_ts = query_db(f"""
            SELECT DATE_TRUNC('month', sf."InvoiceDate") AS mois,
                   SUM(sf."Revenue") AS ca_mensuel
            FROM sales_facts sf
            WHERE sf."StockCode" = '{selected_id}'
            GROUP BY 1 ORDER BY 1
        """)
        if not sales_ts.empty:
            fig = px.line(sales_ts, x="mois", y="ca_mensuel", markers=True,
                          labels={"mois": "Mois", "ca_mensuel": "CA (EUR)"})
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Pas de donnees de ventes pour ce produit.")

    with col_right:
        st.subheader("Distribution des Notes (1-5)")
        ratings = query_db(f"""
            SELECT rf."Rating" AS rating, COUNT(*) AS nb
            FROM review_facts rf
            JOIN product_mapping pm ON pm."Review_ProductCode" = rf."ProductID"
            WHERE pm."ERP_StockCode" = '{selected_id}'
            GROUP BY rf."Rating" ORDER BY rf."Rating"
        """)
        # Fallback : recherche par mots-clés du nom produit
        if ratings.empty:
            keywords = [w for w in selected_name.replace("'", "").split() if len(w) > 3][:3]
            if keywords:
                like_clause = " OR ".join([f'rf."ReviewText" ILIKE \'%{kw}%\'' for kw in keywords])
                ratings = query_db(f"""
                    SELECT rf."Rating" AS rating, COUNT(*) AS nb
                    FROM review_facts rf
                    WHERE {like_clause}
                    GROUP BY rf."Rating" ORDER BY rf."Rating"
                """)
        if not ratings.empty:
            fig2 = px.bar(ratings, x="rating", y="nb",
                          color="nb", color_continuous_scale="RdYlGn",
                          labels={"rating": "Note", "nb": "Nb avis"})
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Aucun avis pour ce produit.")

    st.subheader("3 Derniers Avis")

    # Niveau 1 : correspondance directe via product_mapping
    reviews = query_db(f"""
        SELECT rf."ReviewText" AS txt, rf."Rating" AS rating,
               rf."Sentiment" AS sentiment
        FROM review_facts rf
        JOIN product_mapping pm ON pm."Review_ProductCode" = rf."ProductID"
        WHERE pm."ERP_StockCode" = '{selected_id}'
        ORDER BY rf."ReviewDate" DESC LIMIT 3
    """)
    reviews_source = "direct"

    # Niveau 2 : recherche par mots-clés dans le texte des avis
    if reviews.empty:
        keywords = [w for w in selected_name.replace("'", "").split() if len(w) > 3][:3]
        if keywords:
            like_clause = " OR ".join([f'rf."ReviewText" ILIKE \'%{kw}%\'' for kw in keywords])
            reviews = query_db(f"""
                SELECT rf."ReviewText" AS txt, rf."Rating" AS rating,
                       rf."Sentiment" AS sentiment
                FROM review_facts rf
                WHERE {like_clause}
                ORDER BY rf."Rating" DESC LIMIT 3
            """)
            reviews_source = "keywords"

    # Niveau 3 : avis récents globaux
    if reviews.empty:
        reviews = query_db("""
            SELECT rf."ReviewText" AS txt, rf."Rating" AS rating,
                   rf."Sentiment" AS sentiment
            FROM review_facts rf
            ORDER BY rf."ReviewDate" DESC LIMIT 3
        """)
        reviews_source = "global"

    if not reviews.empty:
        if reviews_source == "keywords":
            st.caption(f"Avis contenant des mots-cles du produit : {', '.join([w for w in selected_name.split() if len(w)>3][:3])}")
        elif reviews_source == "global":
            st.caption("Aucun avis specifique — affichage des avis recents de la plateforme")
        for _, r in reviews.iterrows():
            sentiment_label = "Positif" if r["sentiment"] == "positive" else ("Negatif" if r["sentiment"] == "negative" else "Neutre")
            st.markdown(f"**{r['rating']:.1f}/5** — *{sentiment_label}*")
            st.write(r['txt'])
            st.divider()
    else:
        st.info("Aucun avis disponible.")


def render_data_quality(query_db):
    st.title("Qualite des Donnees  MDM Coverage")

    dq = query_db("SELECT * FROM v_data_quality")
    if dq.empty:
        st.warning("Lance l ETL pour peupler la base.")
        return

    row  = dq.iloc[0]
    taux = float(row.get("Taux_Couverture_MDM", 0))

    c1, c2, c3, c4 = st.columns(4)
    with c1: _kpi("Produits ERP",      row.get("Nb_Produits_ERP", 0))
    with c2: _kpi("Produits avec avis", row.get("Nb_Produits_Avis", 0))
    with c3: _kpi("Mappings MDM",      row.get("Nb_Mappings", 0))
    with c4: _kpi("Taux MDM",          f"{taux:.1f} %")

    st.subheader("Couverture MDM")
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = taux,
        delta = {"reference": 80},
        gauge = {
            "axis": {"range": [0, 100]},
            "bar":  {"color": "royalblue"},
            "steps": [
                {"range": [0,  50], "color": "#ffcccc"},
                {"range": [50, 80], "color": "#fff3cd"},
                {"range": [80, 100], "color": "#d4edda"},
            ],
            "threshold": {"line": {"color": "green", "width": 4}, "thickness": 0.75, "value": 80},
        },
        title = {"text": "Couverture MDM (%)"},
    ))
    st.plotly_chart(fig_gauge, width='stretch')

    st.subheader("Table PRODUCT_MAPPING (extrait)")
    mapping_df = query_db(
        'SELECT "ERP_StockCode" AS erp_code, "ERP_ProductName" AS produit_erp, '
        '"Review_ProductCode" AS code_avis, "Category" AS categorie, '
        '"GoldenRecordName" AS golden_record FROM product_mapping LIMIT 20'
    )
    if not mapping_df.empty:
        st.dataframe(mapping_df, width='stretch', hide_index=True)

    st.subheader("Couverture par Categorie")
    cat_cov = query_db("""
        SELECT pm."Category"                                                    AS category,
               COUNT(DISTINCT pm."ERP_StockCode")                               AS produits_erp,
               COUNT(DISTINCT rf."ProductID")                                   AS produits_avis,
               ROUND(100.0 * COUNT(DISTINCT rf."ProductID")
                     / NULLIF(COUNT(DISTINCT pm."ERP_StockCode"), 0), 1)        AS couverture_pct
        FROM product_mapping pm
        LEFT JOIN review_facts rf ON rf."ProductID" = pm."Review_ProductCode"
        GROUP BY pm."Category"
        ORDER BY couverture_pct DESC
    """)
    if not cat_cov.empty:
        fig_bar = px.bar(
            cat_cov, x="category", y="couverture_pct",
            color="couverture_pct", color_continuous_scale="Blues",
            title="Taux de couverture MDM par categorie",
            labels={"category": "Categorie", "couverture_pct": "Couverture (%)"},
            text_auto=True,
        )
        fig_bar.update_layout(xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(fig_bar, width='stretch')
        st.dataframe(cat_cov, width='stretch', hide_index=True)

    st.subheader("Strategie MDM")
    from src.etl.mdm_mapping import mdm_strategy_description, MDM_STRATEGY
    st.info(f"Stratégie active : **{MDM_STRATEGY.upper()}** — "
            f"configurable via `MDM_STRATEGY` dans `.env`")
    for k, v in mdm_strategy_description().items():
        label  = ("🟢 " if v.get("active") else "⚪ ") + v['fiabilite'] + " — " + v['name']
        with st.expander(label, expanded=v.get("active", False)):
            st.write(v["description"])
