"""
src/ui/analytics.py
=====================
Modules d'analyse avancée pour SmartShop 360 :

  render_temporal_filters()  — Filtres temporels + évolution CA
  render_geo_map()           — Carte choroplèthe des ventes par pays
  render_product_comparison()— Comparaison côte à côte de 2-3 produits
  render_export()            — Export CSV / Excel par page
  render_scoring()           — Score composite produit (CA + note + tendance)
  render_churn()             — Prédiction churn clients (sklearn)
  render_forecast()          — Prévision ventes (Prophet ou régression linéaire)
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta


# ─────────────────────────────────────────────────────────────
#  FILTRES TEMPORELS
# ─────────────────────────────────────────────────────────────

def render_temporal_filters(query_db):
    st.title(" Analyse Temporelle")

    # Plage de dates
    col1, col2 = st.columns(2)
    with col1:
        date_start = st.date_input("Date début", value=date(2010, 1, 1))
    with col2:
        date_end = st.date_input("Date fin", value=date.today())

    if date_start >= date_end:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    granularity = st.selectbox("Granularité", ["Mensuelle", "Trimestrielle", "Annuelle"])
    trunc_map   = {"Mensuelle": "month", "Trimestrielle": "quarter", "Annuelle": "year"}
    trunc       = trunc_map[granularity]

    # Évolution CA
    ts_data = query_db(f"""
        SELECT DATE_TRUNC('{trunc}', "InvoiceDate") AS periode,
               SUM("Revenue")  AS ca,
               SUM("Quantity") AS qte,
               COUNT(DISTINCT "InvoiceNo") AS nb_factures
        FROM sales_facts
        WHERE "InvoiceDate" BETWEEN '{date_start}' AND '{date_end}'
        GROUP BY 1 ORDER BY 1
    """)

    if ts_data.empty:
        st.info("Aucune donnée pour cette période.")
        return

    # KPIs de la période
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("CA Période",    f"{ts_data['ca'].sum():,.0f} EUR")
    with k2: st.metric("Quantités",     f"{int(ts_data['qte'].sum()):,}")
    with k3: st.metric("Nb Factures",   f"{int(ts_data['nb_factures'].sum()):,}")

    fig = px.line(ts_data, x="periode", y="ca", markers=True,
                  title=f"Évolution du CA ({granularity})",
                  labels={"periode": "Période", "ca": "CA (EUR)"})
    fig.add_bar(x=ts_data["periode"], y=ts_data["qte"], name="Quantités",
                yaxis="y2", opacity=0.3)
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="Quantités"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    # Top produits de la période
    st.subheader(" Top Produits de la Période")
    top_products = query_db(f"""
        SELECT pm."GoldenRecordName" AS produit,
               pm."Category"         AS categorie,
               SUM(sf."Revenue")     AS ca,
               SUM(sf."Quantity")    AS qte
        FROM sales_facts sf
        JOIN product_mapping pm ON pm."ERP_StockCode" = sf."StockCode"
        WHERE sf."InvoiceDate" BETWEEN '{date_start}' AND '{date_end}'
        GROUP BY 1, 2 ORDER BY ca DESC LIMIT 10
    """)
    if not top_products.empty:
        fig2 = px.bar(top_products, x="produit", y="ca", color="categorie",
                      title="Top 10 produits — CA",
                      labels={"produit": "Produit", "ca": "CA (EUR)", "categorie": "Catégorie"})
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, width="stretch")

    _export_widget(ts_data, "ca_temporel")


# ─────────────────────────────────────────────────────────────
#  CARTE GÉOGRAPHIQUE
# ─────────────────────────────────────────────────────────────

def render_geo_map(query_db):
    st.title(" Carte des Ventes par Pays")

    geo_data = query_db("""
        SELECT c."Pays"         AS pays,
               COUNT(DISTINCT sf."InvoiceNo") AS nb_commandes,
               SUM(sf."Revenue")     AS ca,
               COUNT(DISTINCT c."ClientID")   AS nb_clients
        FROM sales_facts sf
        JOIN customers c ON c."ClientID" = sf."CustomerID"
        GROUP BY c."Pays" ORDER BY ca DESC
    """)

    if geo_data.empty:
        st.info("Aucune donnée géographique disponible.")
        return

    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Pays", geo_data["pays"].nunique())
    with k2: st.metric("CA Total", f"{geo_data['ca'].sum():,.0f} EUR")
    with k3: st.metric("Clients", f"{int(geo_data['nb_clients'].sum()):,}")

    metric = st.selectbox("Indicateur", ["ca", "nb_commandes", "nb_clients"])
    labels_map = {"ca": "CA (EUR)", "nb_commandes": "Nb Commandes", "nb_clients": "Nb Clients"}

    fig = px.choropleth(
        geo_data,
        locations       = "pays",
        locationmode    = "country names",
        color           = metric,
        hover_name      = "pays",
        hover_data      = {"ca": ":,.0f", "nb_commandes": True, "nb_clients": True},
        color_continuous_scale = "Blues",
        title           = f"Ventes par pays — {labels_map[metric]}",
        labels          = {metric: labels_map[metric]},
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig, width="stretch")

    st.subheader(" Détail par Pays")
    st.dataframe(
        geo_data.rename(columns={"pays": "Pays", "nb_commandes": "Commandes",
                                  "ca": "CA (EUR)", "nb_clients": "Clients"}),
        width="stretch", hide_index=True,
    )
    _export_widget(geo_data, "ventes_geo")


# ─────────────────────────────────────────────────────────────
#  COMPARAISON PRODUITS
# ─────────────────────────────────────────────────────────────

def render_product_comparison(query_db):
    st.title(" Comparaison Produits")

    products = query_db('SELECT "ProductID" AS pid, "ProductName" AS pname FROM products ORDER BY "ProductName"')
    if products.empty:
        st.warning("Base vide.")
        return

    options = dict(zip(products["pname"], products["pid"]))
    names   = list(options.keys())

    selected = st.multiselect(
        "Sélectionnez 2 à 4 produits à comparer",
        names,
        default=names[:2] if len(names) >= 2 else names,
        max_selections=4,
    )
    if len(selected) < 2:
        st.info("Sélectionnez au moins 2 produits.")
        return

    ids_sql = ", ".join(f"'{options[n]}'" for n in selected)
    kpi_df  = query_db(f"""
        SELECT "ProductName" AS produit, "Category" AS categorie,
               "CA" AS ca, "Marge" AS marge, "QuantiteVendue" AS qte,
               "Notemoyenne" AS note, "NbAvis" AS nb_avis,
               "AvisPositifs" AS pos, "AvisNegatifs" AS neg
        FROM v_product_kpi
        WHERE "ProductID" IN ({ids_sql})
    """)

    if kpi_df.empty:
        st.warning("Données insuffisantes.")
        return

    # Tableau de comparaison
    st.subheader(" Tableau de comparaison")
    # Forcer tout en str pour éviter les colonnes object à type mixte (float NaN + str)
    comparison_display = kpi_df.set_index("produit").T.astype(str)
    st.dataframe(comparison_display, width="stretch")

    # Radar chart
    st.subheader(" Radar des métriques normalisées")
    metrics = ["ca", "marge", "qte", "note", "nb_avis"]
    norm_df = kpi_df.copy()
    for m in metrics:
        max_v = norm_df[m].max()
        norm_df[m] = norm_df[m] / max_v if max_v > 0 else 0

    fig = go.Figure()
    for _, row in norm_df.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        fig.add_trace(go.Scatterpolar(
            r    = vals,
            theta = metrics + [metrics[0]],
            fill = "toself",
            name = row["produit"][:30],
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, width="stretch")

    # Bar chart comparatif
    c1, c2 = st.columns(2)
    with c1:
        fig_ca = px.bar(kpi_df, x="produit", y="ca", color="produit",
                        title="CA (EUR)", labels={"ca": "CA (EUR)", "produit": ""})
        fig_ca.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig_ca, width="stretch")
    with c2:
        fig_note = px.bar(kpi_df, x="produit", y="note", color="note",
                          color_continuous_scale="RdYlGn", range_color=[1, 5],
                          title="Note Moyenne", labels={"note": "Note /5", "produit": ""})
        fig_note.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig_note, width="stretch")

    _export_widget(kpi_df, "comparaison_produits")


# ─────────────────────────────────────────────────────────────
#  SCORING PRODUIT COMPOSITE
# ─────────────────────────────────────────────────────────────

def render_scoring(query_db):
    st.title(" Scoring Produit Composite")

    st.markdown("""
    **Formule du score :**
    $$Score = 0.40 \\times CA_{norm} + 0.30 \\times Note_{norm} + 0.20 \\times Qte_{norm} + 0.10 \\times (1 - AvisNeg_{norm})$$
    """)

    df = query_db("""
        SELECT "ProductID" AS pid, "ProductName" AS produit, "Category" AS categorie,
               "CA" AS ca, "Notemoyenne" AS note, "QuantiteVendue" AS qte,
               "AvisNegatifs" AS avis_neg, "NbAvis" AS nb_avis
        FROM v_product_kpi WHERE "NbAvis" > 0
    """)

    if df.empty:
        st.warning("Données insuffisantes pour le scoring.")
        return

    # Normalisation min-max
    def norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn) if mx > mn else pd.Series([0.5] * len(series), index=series.index)

    df["score"] = (
        0.40 * norm(df["ca"]) +
        0.30 * norm(df["note"]) +
        0.20 * norm(df["qte"]) +
        0.10 * (1 - norm(df["avis_neg"]))
    ).round(3)

    df["rank"] = df["score"].rank(ascending=False).astype(int)
    df = df.sort_values("score", ascending=False)

    # Top 10 scoring
    fig = px.bar(
        df.head(10), x="produit", y="score", color="score",
        color_continuous_scale="RdYlGn",
        title="Top 10 Produits — Score Composite",
        labels={"produit": "Produit", "score": "Score (0-1)"},
        text=df.head(10)["score"].map(lambda x: f"{x:.3f}"),
    )
    fig.update_layout(xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # Scatter CA vs Note, taille = qte, couleur = score
    fig2 = px.scatter(
        df, x="note", y="ca", size="qte", color="score",
        hover_name="produit", color_continuous_scale="RdYlGn",
        title="CA vs Note (taille = quantité vendue)",
        labels={"note": "Note Moyenne", "ca": "CA (EUR)", "score": "Score"},
    )
    st.plotly_chart(fig2, width="stretch")

    display_df = df[["rank", "produit", "categorie", "score", "ca", "note", "qte"]].rename(columns={
        "rank": "#", "produit": "Produit", "categorie": "Catégorie",
        "score": "Score", "ca": "CA (EUR)", "note": "Note /5", "qte": "Qté"
    })
    st.dataframe(display_df, width="stretch", hide_index=True)
    _export_widget(df, "scoring_produits")


# ─────────────────────────────────────────────────────────────
#  PRÉDICTION CHURN
# ─────────────────────────────────────────────────────────────

def render_churn(query_db):
    st.title(" Prédiction Churn Clients")

    st.info("Modèle RFM (Récence, Fréquence, Montant) + RandomForest sklearn")

    df = query_db("""
        SELECT c."ClientID",
               c."Pays",
               COUNT(DISTINCT sf."InvoiceNo")   AS frequence,
               SUM(sf."Revenue")                AS montant,
               MAX(sf."InvoiceDate")            AS derniere_commande,
               MIN(sf."InvoiceDate")            AS premiere_commande
        FROM customers c
        JOIN sales_facts sf ON sf."CustomerID" = c."ClientID"
        GROUP BY c."ClientID", c."Pays"
        HAVING COUNT(DISTINCT sf."InvoiceNo") >= 1
    """)

    if df.empty or len(df) < 10:
        st.warning("Données insuffisantes pour l'analyse churn.")
        return

    # Calcul RFM
    ref_date = pd.to_datetime(df["derniere_commande"]).max()  # date de référence = dernière vente connue
    df["derniere_commande"] = pd.to_datetime(df["derniere_commande"])
    df["premiere_commande"] = pd.to_datetime(df["premiere_commande"])
    df["recence_jours"]     = (ref_date - df["derniere_commande"]).dt.days
    df["duree_relation"]    = (df["derniere_commande"] - df["premiere_commande"]).dt.days.clip(lower=1)

    # Seuil dynamique : les 30 % les plus inactifs = churned
    # Évite le cas où tous les clients sont > N jours (données historiques fixes)
    seuil_auto  = int(df["recence_jours"].quantile(0.70))
    seuil_defaut = max(30, seuil_auto)

    seuil = st.slider(
        "Seuil d'inactivité (jours) — en dessous = actif, au-dessus = churned",
        min_value=int(df["recence_jours"].min()),
        max_value=int(df["recence_jours"].max()),
        value=seuil_defaut,
        help=f"Valeur automatique au percentile 70% : {seuil_auto} jours",
    )

    df["churn"] = (df["recence_jours"] > seuil).astype(int)
    n_churn = df["churn"].sum()

    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Clients analysés", len(df))
    with k2: st.metric(f"Clients churned (>{seuil}j)", n_churn)
    with k3: st.metric("Taux de churn", f"{n_churn/len(df)*100:.1f}%")

    # Modèle sklearn
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import LabelEncoder

        features = ["recence_jours", "frequence", "montant", "duree_relation"]
        X = df[features].fillna(0)
        y = df["churn"]

        if y.nunique() < 2:
            st.warning(
                f"Toujours une seule classe avec ce seuil ({seuil} jours). "
                "Déplacez le curseur pour équilibrer les classes (cible : 20-80 % de churn)."
            )
            # Afficher quand même la distribution RFM
            fig_rfm = px.histogram(df, x="recence_jours", nbins=30,
                                   title="Distribution de la récence (jours d'inactivité)",
                                   labels={"recence_jours": "Jours d'inactivité"})
            st.plotly_chart(fig_rfm, width="stretch")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            clf.fit(X_train, y_train)

            df["churn_proba"] = clf.predict_proba(X)[:, 1]
            df["risque"] = pd.cut(df["churn_proba"],
                                   bins=[0, 0.33, 0.66, 1.0],
                                   labels=["Faible", "Moyen", "Élevé"])

            # Importance des features
            imp = pd.DataFrame({"feature": features, "importance": clf.feature_importances_})
            fig_imp = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                             orientation="h", title="Importance des features",
                             color="importance", color_continuous_scale="Blues")
            st.plotly_chart(fig_imp, width="stretch")

            # Distribution des probabilités
            fig_hist = px.histogram(df, x="churn_proba", nbins=20, color="risque",
                                    title="Distribution du risque de churn",
                                    labels={"churn_proba": "Probabilité de churn"})
            st.plotly_chart(fig_hist, width="stretch")

            # Top clients à risque
            st.subheader(" Top 20 Clients à Risque Élevé")
            high_risk = df[df["risque"] == "Élevé"].nlargest(20, "churn_proba")[
                ["ClientID", "Pays", "frequence", "montant", "recence_jours", "churn_proba"]
            ].rename(columns={
                "ClientID": "Client", "Pays": "Pays", "frequence": "Nb Cdes",
                "montant": "CA Total", "recence_jours": "Jours Inactif", "churn_proba": "Risque"
            })
            st.dataframe(high_risk, width="stretch", hide_index=True)
            _export_widget(df, "churn_clients")

    except ImportError:
        st.error("scikit-learn requis : `pip install scikit-learn`")


# ─────────────────────────────────────────────────────────────
#  PRÉVISION DES VENTES
# ─────────────────────────────────────────────────────────────

def render_forecast(query_db):
    st.title(" Prévision des Ventes")

    ts_data = query_db("""
        SELECT DATE_TRUNC('month', "InvoiceDate") AS ds,
               SUM("Revenue") AS y
        FROM sales_facts
        GROUP BY 1 ORDER BY 1
    """)

    if ts_data.empty or len(ts_data) < 3:
        st.warning("Données insuffisantes pour la prévision (minimum 3 mois).")
        return

    ts_data["ds"] = pd.to_datetime(ts_data["ds"])
    horizon = st.slider("Horizon de prévision (mois)", min_value=1, max_value=12, value=3)

    # Tentative Prophet, sinon régression linéaire de fallback
    try:
            import warnings
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tools.sm_exceptions import ConvergenceWarning

            n = len(ts_data)
            last_ds   = ts_data["ds"].max()
            future_ts = pd.date_range(last_ds + pd.DateOffset(months=1), periods=horizon, freq="MS")

            # Hiérarchie de configurations : on dégrade si la convergence échoue.
            # "heuristic" est plus robuste que "estimated" sur petits jeux de données.
            configs = []
            if n >= 24:
                configs.append(dict(trend="add", seasonal="add",   seasonal_periods=12))
            if n >= 12:
                configs.append(dict(trend="add", seasonal="mul",   seasonal_periods=12))
            configs.append(dict(trend="add", seasonal=None))          # tendance seule
            configs.append(dict(trend=None,  seasonal=None))          # lissage simple

            hw_result = None
            model_label = "Holt-Winters"

            for cfg in configs:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        hw_result = ExponentialSmoothing(
                            ts_data["y"].values,
                            trend            = cfg["trend"],
                            seasonal         = cfg["seasonal"],
                            seasonal_periods = cfg.get("seasonal_periods"),
                            initialization_method = "heuristic",
                        ).fit(optimized=True, use_brute=True)
                    # Vérifier que la convergence a abouti
                    retvals = getattr(hw_result, "mle_retvals", None)
                    if retvals is not None and not retvals.get("converged", True):
                        hw_result = None
                        continue
                    # Désignation lisible du modèle retenu
                    parts = []
                    if cfg["trend"]:
                        parts.append("tendance")
                    if cfg["seasonal"]:
                        parts.append(f"saisonnalité {cfg['seasonal']}")
                    model_label = "Holt-Winters" + (f" ({', '.join(parts)})" if parts else " (lissage simple)")
                    break
                except Exception:
                    hw_result = None

            if hw_result is None:
                raise ValueError("Aucune configuration Holt-Winters n'a convergé.")

            y_fit  = hw_result.fittedvalues
            y_pred = hw_result.forecast(horizon)

            # Intervalle de confiance approximatif : ±1.96 * RMSE résidus in-sample
            residuals = ts_data["y"].values - y_fit
            rmse      = float(np.sqrt(np.mean(residuals ** 2)))
            y_upper   = y_pred + 1.96 * rmse
            y_lower   = np.maximum(y_pred - 1.96 * rmse, 0)

            fig = go.Figure()
            fig.add_scatter(
                x=ts_data["ds"], y=ts_data["y"],
                mode="markers+lines", name="Données réelles",
                marker=dict(color="royalblue"),
            )
            fig.add_scatter(
                x=ts_data["ds"], y=y_fit,
                mode="lines", name=f"Ajustement {model_label}",
                line=dict(color="green", dash="dot"),
            )
            fig.add_traces([
                go.Scatter(
                    x=future_ts, y=y_upper,
                    fill=None, mode="lines",
                    line=dict(width=0), showlegend=False,
                ),
                go.Scatter(
                    x=future_ts, y=y_lower,
                    fill="tonexty", mode="lines",
                    line=dict(width=0),
                    name="Intervalle de confiance 95%",
                    fillcolor="rgba(255,165,0,0.2)",
                ),
            ])
            fig.add_scatter(
                x=future_ts, y=y_pred,
                mode="lines+markers", name="Prévision",
                line=dict(color="orange", dash="dash"),
            )
            fig.update_layout(
                title=f"Prévision CA — {horizon} mois ({model_label})",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")

            prev_df = pd.DataFrame({
                "Mois":        future_ts.strftime("%Y-%m"),
                "CA Prévu":    y_pred.round(2),
                "Borne Basse": y_lower.round(2),
                "Borne Haute": y_upper.round(2),
            })
            st.subheader("Prévision détaillée")
            st.dataframe(prev_df, width="stretch", hide_index=True)
            _export_widget(prev_df, "prevision_ventes")

    except (ImportError, ValueError):
            # Fallback 2 : régression linéaire (dernier recours)
            st.info(
                "régression linéaire utilisée en dernier recours.\n\n"
                "`pip install statsmodels` pour Holt-Winters, "
                "`pip install prophet` pour Prophet."
            )
            from sklearn.linear_model import LinearRegression

            ts_data["t"] = np.arange(len(ts_data))
            lr = LinearRegression()
            lr.fit(ts_data[["t"]], ts_data["y"])

            last_ds   = ts_data["ds"].max()
            future_ts = pd.date_range(last_ds + pd.DateOffset(months=1), periods=horizon, freq="MS")
            future_t  = pd.DataFrame({"t": np.arange(len(ts_data), len(ts_data) + horizon)})
            y_pred    = lr.predict(future_t)

            fig = go.Figure()
            fig.add_scatter(
                x=ts_data["ds"], y=ts_data["y"],
                mode="markers+lines", name="Données réelles",
                marker=dict(color="royalblue"),
            )
            fig.add_scatter(
                x=future_ts, y=y_pred,
                mode="lines+markers", name="Prévision linéaire",
                line=dict(color="orange", dash="dash"),
            )
            fig.update_layout(title=f"Prévision CA — {horizon} mois (Régression linéaire)")
            st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────
#  EXPORT WIDGET (réutilisable)
# ─────────────────────────────────────────────────────────────

def _export_widget(df: pd.DataFrame, filename_prefix: str):
    """Affiche des boutons d'export CSV et Excel."""
    if df.empty:
        return

    st.markdown("---")
    c1, c2, _ = st.columns([1, 1, 4])

    # CSV
    with c1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label    = " CSV",
            data     = csv,
            file_name = f"{filename_prefix}.csv",
            mime     = "text/csv",
        )

    # Excel
    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        buf.seek(0)
        st.download_button(
            label    = " Excel",
            data     = buf.read(),
            file_name = f"{filename_prefix}.xlsx",
            mime     = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
