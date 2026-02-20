# SmartShop 360 — Livrables Urbanistiques & Architecture

> **Document de référence** — Modélisation du SI, diagrammes techniques, stratégie MDM
> et dictionnaire de données.

---

## Table des matières

1. [Vue Métier (ArchiMate)](#1-vue-métier)
2. [Vue Transformation — SI Opérationnel → MDM → SI Décisionnel](#2-vue-transformation)
3. [Diagramme de Flux de Données (DFD) — Pipeline ETL](#3-diagramme-de-flux-de-données-dfd)
4. [Schéma d'Architecture Physique (C4 Level 2)](#4-schéma-darchitecture-physique)
5. [Stratégie MDM — Choix et Justification](#5-stratégie-mdm)
6. [Dictionnaire de Données](#6-dictionnaire-de-données)

---

## 1. Vue Métier

> **Style ArchiMate** — Processus existants, acteurs et flux métier inter-silos.

```mermaid
flowchart TD
    subgraph ACTEURS["Acteurs Métier"]
        direction LR
        A1["Acheteur / Client"]
        A2["Gestionnaire\nde Stock (ERP)"]
        A3["Analyste Data"]
        A4["Responsable\nMarketing"]
        A5["Agent IA\n(SmartShop 360)"]
    end

    subgraph P1["Processus : Vente sur Site Web"]
        direction TB
        P1a["Parcourir\nle Catalogue"]
        P1b["Passer\nCommande"]
        P1c["Enregistrer\nTransaction ERP"]
        P1a --> P1b --> P1c
    end

    subgraph P2["Processus : Vente Marketplace"]
        direction TB
        P2a["Consultation\nProduit"]
        P2b["Achat\nMarketplace"]
        P2c["Dépôt Avis\n& Note Client"]
        P2a --> P2b --> P2c
    end

    subgraph P3["Processus : Analyse Qualité (SI Décisionnel)"]
        direction TB
        P3a["Unification des\nRéférentiels (MDM)"]
        P3b["Calcul KPIs\nCroisés"]
        P3c["Détection Alertes\n(Produits CRITIQUE)"]
        P3d["Interrogation\nlanguage naturel"]
        P3a --> P3b --> P3c
        P3b --> P3d
    end

    A1 -->|"navigue"| P1a
    A1 -->|"achète"| P2a
    A1 -->|"dépose avis"| P2c
    A2 -->|"gère stock"| P1c
    A3 -->|"pilote ETL"| P3a
    A3 -->|"analyse"| P3b
    A4 -->|"consulte alertes"| P3c
    A5 -->|"répond"| P3d

    P1c -->|"flux CSV (ERP)"| P3a
    P2c -->|"flux JSON (HuggingFace)"| P3a

    style ACTEURS fill:#E8F4FD,stroke:#2980B9
    style P1 fill:#EBF5EB,stroke:#27AE60
    style P2 fill:#FEF9E7,stroke:#F39C12
    style P3 fill:#F5EEF8,stroke:#8E44AD
```

### Lecture

| Acteur | Rôle dans la chaîne de valeur |
|--------|-------------------------------|
| **Acheteur / Client** | Génère les transactions ERP et les avis marketplace |
| **Gestionnaire de Stock** | Saisit et valide les données dans l'ERP |
| **Analyste Data** | Pilote le pipeline ETL, supervise la qualité MDM |
| **Responsable Marketing** | Consomme les alertes produits CRITIQUE (Note < 3.0 ET Qté > 50) |
| **Agent IA** | Répond aux questions en langage naturel via Text-to-SQL |

---

## 2. Vue Transformation

> **SI Opérationnel → Référentiel Produit MDM → SI Décisionnel**
> Le composant MDM est le **pivot** qui brise les deux silos sans identifiant commun.

```mermaid
flowchart LR
    subgraph SIO["SI OPERATIONNEL — Sources"]
        direction TB
        ERP["ERP\nonline_retail_II.csv\n1 067 371 transactions\n(Kaggle — 2009-2011)"]
        WEB["Web / Marketplace\nlabeledReview.json\n1 000 avis clients\n(HuggingFace)"]
    end

    subgraph MDM["REFERENTIEL PRODUIT — Pivot MDM"]
        direction TB
        MDM1["Nettoyage &\nNormalisation"]
        MDM2["Matching TF-IDF\n(char_wb ngram 2-4\ncosine >= 0.35)"]
        MDM3["Golden Record\nproduct_mapping\n9 colonnes"]
        MDM1 --> MDM2 --> MDM3
    end

    subgraph SID["SI DECISIONNEL — Data Warehouse PostgreSQL 16"]
        direction TB
        T1[("products\n(Référentiel unifié)")]
        T2[("sales_facts\n(Transactions enrichies)")]
        T3[("review_facts\n(Avis avec sentiment)")]
        T4[("review_embeddings\n384 dim pgvector)")]
        T5[("conversations\n(Mémoire Agent)")]
        V1["v_product_kpis\nv_alerts\nv_monthly_revenue\nv_data_quality"]
        T1 --- T2
        T1 --- T3
        T3 --- T4
        T2 & T3 --> V1
    end

    subgraph APP["APPLICATION — Couches Consommatrices"]
        UI["Streamlit UI\n10 écrans analytiques\n(Plotly / Pandas)"]
        AG["Agent Text-to-SQL\nmulti-LLM\n(Groq / Mistral / OpenAI)"]
        ML["ML Layer\nChurn RFM\nForecast Holt-Winters\nScoring"]
        OBS["Observabilité\nPrometheus + Grafana\nAlertes email/Slack"]
    end

    ERP -->|"SHA-256 incrémental"| MDM1
    WEB -->|"SHA-256 incrémental"| MDM1
    MDM3 -->|"INSERT"| T1
    ERP  -->|"clean_transactions()"| T2
    WEB  -->|"clean_reviews()"| T3
    T3   -->|"sentence-transformers\nall-MiniLM-L6-v2"| T4

    SID --> UI
    SID --> AG
    SID --> ML
    SID --> OBS

    style SIO  fill:#FDEBD0,stroke:#E67E22,color:#000
    style MDM  fill:#D5F5E3,stroke:#1E8449,color:#000
    style SID  fill:#D6EAF8,stroke:#1A5276,color:#000
    style APP  fill:#F4ECF7,stroke:#76448A,color:#000
```

### Séparation SI Opérationnel / SI Décisionnel

| Dimension | SI Opérationnel | SI Décisionnel |
|-----------|----------------|----------------|
| **Rôle** | Capturer les faits bruts | Analyser, croiser, alerter |
| **Sources** | CSV Kaggle + JSON HuggingFace | PostgreSQL 16 (Data Warehouse) |
| **Granularité** | Ligne de commande / Avis individuel | Agrégats mensuels, scores produits |
| **Latence** | Temps réel / batch | Batch ETL (incrémental SHA-256) |
| **Pivot MDM** | Aucun identifiant commun | `product_mapping` — Golden Record |

---

## 3. Diagramme de Flux de Données (DFD)

> **Pipeline ETL complet** : Source → Nettoyage → Qualité → Mapping/Unification → Base Finale.

```mermaid
flowchart TD
    subgraph SRC["SOURCES (Externes)"]
        S1[/"online_retail_II.csv\n1 067 371 lignes"/]
        S2[/"labeledReview.json\n1 000 avis"/]
    end

    subgraph HASH["CONTROLE INCREMENTAL"]
        H1{"SHA-256\nchangé ?"}
        H2(["Skip ETL\n(.etl_hashes.json)"])
    end

    subgraph CLEAN["1 — EXTRACTION & NETTOYAGE\n(src/etl/cleaning.py)"]
        C1["load_transactions()\nEchantillon 30 000 / SEED=42"]
        C2["clean_transactions()\n- Filtre factures C (retours)\n- Qty > 0, Price > 0\n- CustomerID non-null\n- Revenue = Qty x Price\n- Margin = Revenue x taux(20-45%)"]
        C3["load_reviews()\nNormalisation sentiment\n(pos/neg -> 1.0/0.0)"]
        C4["extract_top50_products()"]
    end

    subgraph DQ["2 — QUALITE DES DONNEES\n(src/etl/data_quality.py)"]
        D1["DataQualityReport\n7 règles validées :\n- pas de nulls clés\n- Qty/Price > 0\n- Revenue cohérent\n- format InvoiceDate\n- CustomerID valide"]
    end

    subgraph MDM["3 — MAPPING & UNIFICATION\n(src/etl/mdm_mapping.py)"]
        M1["build_product_mapping()\nTF-IDF char_wb ngram(2,4)\nCosine similarity"]
        M2{"Score\n>= 0.35 ?"}
        M3["Match confirme\nERP_StockCode lie\nau ReviewTitle"]
        M4["Fallback rank-based\nTop-N par frequence"]
        M5["infer_category()\nMots-cles -> 6 categories + Divers"]
        M1 --> M2
        M2 -->|"oui"| M3
        M2 -->|"non"| M4
        M3 & M4 --> M5
    end

    subgraph LOAD["4 — CHARGEMENT BASE FINALE\n(src/etl/run_etl.py)"]
        L1[("products\nGolden Record MDM")]
        L2[("sales_facts\nTransactions enrichies")]
        L3[("review_facts\nAvis avec sentiment")]
        L4[("review_embeddings\n384 dim pgvector")]
        L5["4 Vues SQL analytiques"]
        L1 & L2 & L3 --> L5
        L3 --> L4
    end

    S1 & S2 --> H1
    H1 -->|"non change"| H2
    H1 -->|"change"| C1
    S1 --> C1
    S2 --> C3
    C1 --> C2 --> C4 --> D1
    C3 --> D1
    D1 --> M1
    M5 --> L1
    C2 --> L2
    C3 --> L3

    style SRC fill:#FDEBD0,stroke:#E67E22
    style HASH fill:#EAFAF1,stroke:#1E8449
    style CLEAN fill:#EBF5FB,stroke:#2E86C1
    style DQ fill:#FEF9E7,stroke:#F39C12
    style MDM fill:#F4ECF7,stroke:#8E44AD
    style LOAD fill:#EBF5EB,stroke:#27AE60
```

### Détail des étapes

| Étape | Fichier | Entrée | Sortie | Règles métier clés |
|-------|---------|--------|--------|--------------------|
| **Extraction** | `cleaning.py` | CSV brut (1 M lignes) | DataFrame 30 000 lignes (SEED=42) | Échantillon reproductible |
| **Nettoyage transactions** | `cleaning.py` | DataFrame brut | DataFrame propre | Filtre factures `C`, Qty/Price > 0, suppression CustomerID null |
| **Nettoyage avis** | `cleaning.py` | JSON HuggingFace | DataFrame sentiment normalisé | `positive` → 1.0, `negative` → 0.0 |
| **Qualité données** | `data_quality.py` | DataFrames nettoyés | `DataQualityReport` (7 règles) | Validation pré-chargement |
| **MDM Matching** | `mdm_mapping.py` | Top-50 produits ERP + titres avis | `product_mapping` (9 colonnes) | TF-IDF cosine ≥ 0.35 / fallback rank |
| **Chargement** | `run_etl.py` | DataFrames validés | 6 tables + 4 vues PostgreSQL | DDL idempotent (`CREATE TABLE IF NOT EXISTS`) |

---

## 4. Schéma d'Architecture Physique

> **Style C4 Level 2 (Conteneurs)** — Infrastructure Docker, flux réseau, API externes.

```mermaid
C4Context
    title Architecture Physique — SmartShop 360

    Person(user, "Utilisateur", "Analyste / Data Steward — navigateur web")

    System_Boundary(docker, "Docker Compose (localhost)") {
        Container(app, "smartshop360_app", "Python 3.11 / Streamlit", "10 écrans analytiques — Agent Text-to-SQL — ML (sklearn, statsmodels) — Prometheus :8000")
        ContainerDb(db, "smartshop360_db", "PostgreSQL 16 + pgvector", "6 tables — 4 vues — pool_size=5, pool_recycle=1800 — statement_timeout=30s — port 5432")
    }

    System_Ext(llm_groq, "Groq API", "llama-3.3-70B")
    System_Ext(llm_mistral, "Mistral API", "mistral-large")
    System_Ext(llm_openai, "OpenAI API", "gpt-4o")
    System_Ext(smtp, "SMTP / Slack", "Alertes email et webhook")

    System_Boundary(mon, "Monitoring (optionnel)") {
        Container(prom, "Prometheus :9090", "Scrape métriques", "etl_runs_total — agent_requests_total")
        Container(graf, "Grafana :3000", "Dashboards", "6 panneaux — 4 règles d alerte")
    }

    Rel(user, app, "HTTPS / :8501", "navigateur")
    Rel(app, db, "SQLAlchemy pool", "TCP :5432")
    Rel(app, llm_groq, "REST HTTPS", "Text-to-SQL priorité 1")
    Rel(app, llm_mistral, "REST HTTPS", "Fallback priorité 2")
    Rel(app, llm_openai, "REST HTTPS", "Fallback priorité 3")
    Rel(app, smtp, "SMTP/Webhook", "alertes CRITIQUE")
    Rel(prom, app, "scrape :8000/metrics", "Prometheus pull")
    Rel(graf, prom, "PromQL", "visualisation")
```

### Ressources et ports

| Conteneur | Image | Port exposé | Volume |
|-----------|-------|-------------|--------|
| `smartshop360_db` | `postgres:16-alpine` | `5432` | `pgdata` (persistant) |
| `smartshop360_app` | `Dockerfile` (Python 3.11-slim) | `8501` | `./Data:/app/Data` (ro) |
| `prometheus` | `prom/prometheus` | `9090` | — |
| `grafana` | `grafana/grafana` | `3000` | — |

### Résilience de l'Agent Text-to-SQL

La chaîne de fallback garantit que l'application reste opérationnelle sans clé API :

```
Groq (llama-3.3-70B) → Mistral (mistral-large) → OpenAI (gpt-4o) → Anthropic → Fallback SQL (règles)
```

---

## 5. Stratégie MDM

### 5.1 Problématique

Les deux sources de données (**ERP** et **Marketplace**) ne partagent aucun identifiant commun :

- L'ERP identifie les produits par un `StockCode` alphanumérique (ex. `85123A`)
- Les avis clients référencent les produits par leur **nom textuel** (ex. `"WHITE HANGING HEART T-LIGHT HOLDER"`)

**Conséquence** : impossible de croiser directement « CA par produit » et « note moyenne par produit » sans une couche de réconciliation.

### 5.2 Implémentation POC (ce projet)

#### Niveau 1 — TF-IDF sur n-grammes de caractères

```python
# src/etl/mdm_mapping.py
vectorizer = TfidfVectorizer(
    analyzer="char_wb",   # n-grammes de caractères (robuste aux abréviations)
    ngram_range=(2, 4),   # bigrammes à 4-grammes
    min_df=1,
)
matrix = vectorizer.fit_transform(all_names)
similarity = cosine_similarity(erp_vectors, review_vectors)
```

**Seuil de décision** : `cosine_similarity ≥ 0.35`
- Au-dessus : **match confirmé** → `ERP_StockCode` lié au `ReviewProductTitle`
- En-dessous : **fallback rank-based** → on associe le N-ième produit le plus fréquent

#### Niveau 2 — Inférence de catégorie

```python
_CATEGORY_RULES = [
    (["bag", "tote", "purse"],          "Sacs & Accessoires"),
    (["candle", "light", "lamp"],       "Bougies & Luminaires"),
    (["mug", "cup", "tea", "coffee"],   "Cuisine & Table"),
    (["frame", "sign", "card"],         "Papeterie & Décoration"),
    (["christmas", "xmas", "santa"],    "Fêtes & Saisonniers"),
    (["garden", "plant", "pot"],        "Jardin & Extérieur"),
]
# Fallback : "Divers"
```

#### Résultat : table `product_mapping` (Golden Record)

| Colonne | Description |
|---------|-------------|
| `ERP_StockCode` | Identifiant ERP (clé primaire de réconciliation) |
| `ERP_ProductName` | Libellé normalisé côté ERP |
| `ReviewProductTitle` | Libellé côté avis client |
| `MatchScore` | Score cosinus TF-IDF (0.0–1.0) |
| `MatchMethod` | `tfidf` ou `rank_fallback` |
| `Category` | Catégorie inférée par mots-clés |
| `MasterProductName` | Libellé Gold Record (ERP par défaut) |
| `IsGoldenRecord` | Booléen — validé par le Data Steward |
| `CreatedAt` | Horodatage du Golden Record |

### 5.3 Approche en contexte réel (Production)

Dans un SI de production, la stratégie MDM s'organise en **4 niveaux de confiance** :

| Niveau | Méthode | Confiance | Action |
|--------|---------|-----------|--------|
| **1 — Exact** | Code EAN/GTIN commun aux deux systèmes | 100 % | Intégration automatique |
| **2 — Phonétique/Token** | Levenshtein + TF-IDF hybride (`0.6 × cosine + 0.4 × levenshtein_norm`) | Haute (≥ 0.70) | Intégration automatique |
| **3 — Sémantique** | Sentence-BERT (`all-MiniLM-L6-v2`) — similarité cosinus sur embeddings 384d | Moyenne (0.50–0.70) | Validation humaine (Data Steward) |
| **4 — Rank fallback** | Fréquence de co-occurrence dans les logs | Basse (< 0.50) | Mise en quarantaine + workflow de dérogation |

**Score hybride Production** :

$$S_{hybride} = 0.6 \times \cos(\vec{tfidf}_A, \vec{tfidf}_B) + 0.4 \times \frac{1 - d_{Levenshtein}(A, B)}{\max(|A|,|B|)}$$

**Gouvernance recommandée** :

- **Data Steward** : valide les enregistrements de niveau 3 via une interface de revue
- **Workflow de dérogation** : les produits niveau 4 sont signalés dans `v_data_quality` pour action manuelle
- **Golden Record versionné** : chaque modification de `product_mapping` est horodatée et conservée dans une table d'historique (`product_mapping_history`)
- **Synchronisation ERP** : idéalement, l'ERP émet un code EAN/GTIN à la création produit — ce code devient la clé de réconciliation universelle

---

## 6. Dictionnaire de Données

> Schéma en étoile simplifié :
> **`products`** (dimension centrale) ← `sales_facts` + `review_facts` (faits)

```mermaid
erDiagram
    products {
        TEXT stock_code PK
        TEXT description
        TEXT category
        FLOAT unit_price
        INT total_quantity
        FLOAT total_revenue
        FLOAT avg_review_score
        INT review_count
        TIMESTAMPTZ created_at
    }

    sales_facts {
        TEXT invoice_no
        TEXT stock_code FK
        TEXT description
        INT quantity
        TIMESTAMPTZ invoice_date
        FLOAT unit_price
        TEXT customer_id
        TEXT country
        FLOAT revenue
        FLOAT margin
    }

    review_facts {
        SERIAL id PK
        TEXT product_title
        TEXT stock_code FK
        FLOAT rating
        FLOAT sentiment_score
        TEXT sentiment_label
        TEXT review_text
        TIMESTAMPTZ review_date
    }

    review_embeddings {
        INT review_id FK
        VECTOR embedding
    }

    conversations {
        SERIAL id PK
        TEXT session_id
        TEXT role
        TEXT content
        TIMESTAMPTZ created_at
    }

    products ||--o{ sales_facts : "stock_code"
    products ||--o{ review_facts : "stock_code"
    review_facts ||--o| review_embeddings : "id"
```

### Table `products` — Référentiel Produit MDM (Dimension)

| Colonne | Type SQL | Nullable | Description |
|---------|----------|----------|-------------|
| `stock_code` | `TEXT` | NON | Identifiant ERP unique — clé primaire |
| `description` | `TEXT` | OUI | Libellé Gold Record (Master Product Name) |
| `category` | `TEXT` | OUI | Catégorie inférée par `infer_category()` |
| `unit_price` | `FLOAT` | OUI | Prix unitaire de référence (GBP) |
| `total_quantity` | `INT` | OUI | Quantité totale vendue sur la période |
| `total_revenue` | `FLOAT` | OUI | Chiffre d'affaires total (GBP) |
| `avg_review_score` | `FLOAT` | OUI | Note moyenne des avis (0.0–5.0) |
| `review_count` | `INT` | OUI | Nombre d'avis associés |
| `created_at` | `TIMESTAMPTZ` | OUI | Date de création du Golden Record |

**Clé de réconciliation MDM** : `stock_code` est issu de la table `product_mapping` — c'est le seul identifiant commun entre `sales_facts` et `review_facts` après la phase de matching TF-IDF.

---

### Table `sales_facts` — Faits de Vente (Source ERP)

| Colonne | Type SQL | Nullable | Description |
|---------|----------|----------|-------------|
| `invoice_no` | `TEXT` | NON | Numéro de facture ERP |
| `stock_code` | `TEXT` | NON | FK → `products.stock_code` |
| `description` | `TEXT` | OUI | Libellé produit tel que saisi dans l'ERP |
| `quantity` | `INT` | NON | Quantité commandée (toujours > 0 après nettoyage) |
| `invoice_date` | `TIMESTAMPTZ` | NON | Horodatage de la transaction |
| `unit_price` | `FLOAT` | NON | Prix unitaire au moment de la vente (GBP) |
| `customer_id` | `TEXT` | NON | Identifiant client (non null après nettoyage) |
| `country` | `TEXT` | OUI | Pays de livraison (pour vue géographique) |
| `revenue` | `FLOAT` | NON | `quantity × unit_price` — calculé à l'ETL |
| `margin` | `FLOAT` | NON | `revenue × taux_marge` (taux aléatoire 20–45 %, simulé) |

**Règles de nettoyage** :
- Les factures dont le `invoice_no` commence par `C` (avoir/retour) sont supprimées
- `quantity > 0` et `unit_price > 0` — les anomalies sont rejetées
- `customer_id` null → ligne rejetée (impossibilité d'analyse RFM)

---

### Table `review_facts` — Faits Avis Clients (Source Marketplace)

| Colonne | Type SQL | Nullable | Description |
|---------|----------|----------|-------------|
| `id` | `SERIAL` | NON | Clé primaire auto-incrémentée |
| `product_title` | `TEXT` | NON | Titre produit tel que rédigé dans l'avis |
| `stock_code` | `TEXT` | OUI | FK → `products.stock_code` (null si non reconcilié) |
| `rating` | `FLOAT` | OUI | Note sur 5 étoiles (0.0–5.0) |
| `sentiment_score` | `FLOAT` | OUI | Score normalisé : 1.0 = positif, 0.0 = négatif |
| `sentiment_label` | `TEXT` | OUI | Label brut : `positive` ou `negative` |
| `review_text` | `TEXT` | OUI | Texte complet de l'avis |
| `review_date` | `TIMESTAMPTZ` | OUI | Date de publication de l'avis |

**Note** : `stock_code` peut être null si le produit de l'avis n'a pas trouvé de match TF-IDF ≥ 0.35 et que le fallback rank-based n'a pas pu s'appliquer. Ces lignes sont comptabilisées dans la vue `v_data_quality`.

---

### Table `review_embeddings` — Vecteurs Sémantiques (RAG)

| Colonne | Type SQL | Nullable | Description |
|---------|----------|----------|-------------|
| `review_id` | `INT` | NON | FK → `review_facts.id` |
| `embedding` | `VECTOR(384)` | NON | Embedding `all-MiniLM-L6-v2` (pgvector) |

**Usage** : Recherche ANN (`ivfflat` index, `lists=100`) pour le RAG de l'Agent IA.
La requête `ReviewRAG.search("probleme emballage", k=5)` retourne les 5 avis sémantiquement les plus proches pour contextualiser la réponse LLM.

---

### Vues SQL analytiques

| Vue | Jointures | Utilisation |
|-----|-----------|-------------|
| `v_product_kpis` | `products ⋈ sales_facts ⋈ review_facts` | Écran "Analyse Produit" — fiche 360° |
| `v_alerts` | `v_product_kpis` (filtre Note < 3.0 AND Qté > 50) | Écran "Dashboard" — alertes CRITIQUE |
| `v_monthly_revenue` | `sales_facts` (GROUP BY mois) | Écran "Analyse Temporelle" — prévisions |
| `v_data_quality` | `products` (comptes null, taux couverture MDM) | Écran "Qualité des Données" |

---

## Synthèse des choix techniques

| Composant | Choix POC | Justification | Alternative Production |
|-----------|-----------|---------------|------------------------|
| **Stockage** | PostgreSQL 16 + pgvector | ACID, vues SQL complexes, pgvector pour RAG | Idem + partitionnement par date |
| **MDM Matching** | TF-IDF cosine (char_wb, ngram 2-4) | Robuste aux abréviations et fautes, pas de GPU requis | TF-IDF + Levenshtein hybride + Sentence-BERT validation |
| **Réconciliation clés** | Nom produit textuel → score cosinus | Seule option sans code commun EAN/GTIN | EAN/GTIN en priorité 1, puis hybride |
| **Prévisions** | Holt-Winters (ExponentialSmoothing) | Capture tendance + saisonnalité, fallback gracieux | Prophet (si données > 2 ans) |
| **Agent IA** | Multi-LLM avec fallback SQL | Fonctionnel sans clé API (démo) | LangGraph + RAG structuré |
| **Orchestration** | `app.py` init_db → run_etl auto | Démarrage en 1 commande | Airflow / Prefect pour batch planifié |
| **Monitoring** | Prometheus + Grafana (optionnel) | Observable en production, stack standard | Idem + alerting PagerDuty |
