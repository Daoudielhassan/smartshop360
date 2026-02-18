from src.db_config import get_engine
import pandas as pd

engine = get_engine()

queries = {
    "CA by category": 'SELECT "Category", SUM("CA") AS "CA" FROM v_product_kpi GROUP BY "Category" ORDER BY "CA" DESC',
    "Couverture_pct": '''
        SELECT pm."Category",
            COUNT(DISTINCT pm."ERP_StockCode") AS "Produits_ERP",
            COUNT(DISTINCT rf."ProductID")     AS "Produits_Avis",
            ROUND(100.0 * COUNT(DISTINCT rf."ProductID")
                  / NULLIF(COUNT(DISTINCT pm."ERP_StockCode"), 0), 1) AS "Couverture_pct"
        FROM product_mapping pm
        LEFT JOIN review_facts rf ON rf."ProductID" = pm."Review_ProductCode"
        GROUP BY pm."Category"
        ORDER BY "Couverture_pct" DESC
    ''',
    "Global KPIs": 'SELECT SUM("CA") AS ca_total, AVG("Notemoyenne") AS note_moy, SUM("NbAvis") AS nb_avis, COUNT(*) AS nb_produits FROM v_product_kpi',
}

for name, sql in queries.items():
    df = pd.read_sql(sql, engine)
    print(f"\n[{name}]")
    print(f"  cols: {df.columns.tolist()}")
    print(f"  rows: {len(df)}")
    if not df.empty:
        print(f"  sample: {df.iloc[0].to_dict()}")
