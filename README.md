# SmartShop 360 - Plateforme Data & Agent IA E-commerce

---

## Table des Matieres

1. [Vision du Projet](#1-vision-du-projet)
2. [Architecture Generale](#2-architecture-generale)
3. [Choix Technologiques Justifies](#3-choix-technologiques-justifies)
4. [Structure du Projet](#4-structure-du-projet)
5. [Demarrage Rapide](#5-demarrage-rapide)
6. [Pipeline ETL - Detail Complet](#6-pipeline-etl---detail-complet)
7. [Schema de la Base de Donnees](#7-schema-de-la-base-de-donnees)
8. [Strategie MDM et Matching Produit](#8-strategie-mdm-et-matching-produit)
9. [Qualite des Donnees](#9-qualite-des-donnees)
10. [Agent IA Text-to-SQL](#10-agent-ia-text-to-sql)
11. [Memoire et RAG](#11-memoire-et-rag)
12. [Fonctionnalites par Ecran](#12-fonctionnalites-par-ecran)
13. [Machine Learning](#13-machine-learning)
14. [Observabilite et Alertes](#14-observabilite-et-alertes)
15. [Tests et CI/CD](#15-tests-et-cicd)
16. [Deploiement Docker](#16-deploiement-docker)
17. [KPIs du Projet](#17-kpis-du-projet)
18. [Exemples de Questions pour l'Agent](#18-exemples-de-questions-pour-lagent)

---

## 1. Vision du Projet

### 1.1 Problematique Metier

Un e-commercant B2C specialise en Decoration et Cadeaux souffre d'une fragmentation de son systeme
d'information en deux silos independants :

- **Silo Ventes / Finance** : l'ERP centralise les transactions (factures, produits vendus, CA, marges).
  L'equipe sait ce qui se vend, mais pas pourquoi certains produits perdent en attractivite.

- **Silo Marketing / Qualite** : les avis clients sur les marketplaces revelent la reputation produit.
  L'equipe sait ce que les clients pensent, mais ne peut pas croiser ces donnees avec les chiffres de vente.

**Consequence directe** : impossibilite de repondre a des questions metier strategiques telles que :

  - "Quels sont nos best-sellers qui commencent a avoir une mauvaise reputation ?"
  - "Quels produits ont un fort volume de ventes mais des avis majoritairement negatifs ?"
  - "Quels clients risquent de partir (churn) et quels produits les ont decus ?"

### 1.2 Solution Cible

SmartShop 360 est une plateforme Data & IA qui brise ces silos en :

1. **Unifiant** les deux sources dans un entrepot PostgreSQL commun via un pipeline ETL robuste.
2. **Reconciliant** les referentiels produits (MDM) malgre l'absence d'identifiant commun.
3. **Exposant** les KPIs croises via 10 ecrans analytiques Streamlit.
4. **Permettant** des analyses en langage naturel via un agent Text-to-SQL multi-LLM.

### 1.3 Sources de Donnees

| Source                                   | Type           | Description                                    |
|------------------------------------------|----------------|------------------------------------------------|
| Online Retail II (Kaggle)                | CSV structure  | 1 067 371 transactions reelles (2009-2011)     |
| ecommerce-product-reviews (HuggingFace)  | JSON labelise  | 1 000 avis clients avec sentiment et note      |

Les deux datasets sont des donnees reelles publiques qui simulent respectivement l'ERP
(transactions) et le flux d'avis marketplace. Ils n'ont aucun identifiant commun, ce qui
rend le probleme MDM (Master Data Management) realiste et non-trivial.

---

## 2. Architecture Generale

### 2.1 Vue d'Ensemble

```
Sources de donnees
      |
      v
+---------------------+
|  Pipeline ETL       |  src/etl/
|  - cleaning.py      |  Nettoyage, echantillonnage, calcul Revenue/Margin
|  - incremental.py   |  Detection SHA-256 (evite les rechargements inutiles)
|  - data_quality.py  |  10+ regles de validation (Great Expectations-style)
|  - mdm_mapping.py   |  Reconciliation produits ERP <-> Avis (TF-IDF)
|  - run_etl.py       |  Orchestrateur + DDL PostgreSQL
+---------------------+
      |
      v
+---------------------+
|  PostgreSQL 16      |  6 tables + 4 vues analytiques
|  (pgvector active)  |  Stockage unifie + index vectoriels (RAG)
+---------------------+
      |
      +------------------+------------------+
      v                  v                  v
+------------+   +---------------+   +-------------+
|  UI Layer  |   |  Agent Layer  |   |  ML Layer   |
| (Streamlit)|   | (Text-to-SQL) |   | (sklearn)   |
| 10 ecrans  |   | multi-LLM     |   | churn       |
| Plotly     |   | memoire PG    |   | scoring     |
|            |   | RAG pgvector  |   | forecast    |
+------------+   +---------------+   +-------------+
      |
      v
+---------------------+
|  Observabilite      |  monitoring/
|  Prometheus         |  metriques ETL, agent, ML
|  Grafana            |  dashboard 6 panneaux
|  Alertes email/Slack|  produits CRITIQUE
+---------------------+
```

### 2.2 Flux de Donnees

Au premier lancement de `streamlit run app.py` :

1. `app.py` appelle `init_db()` (mis en cache par `@st.cache_resource`).
2. `init_db()` verifie si la table `products` est vide.
3. Si vide, l'ETL est lance avec `run_etl(force=True)`.
4. L'ETL charge 6 tables et cree 4 vues SQL analytiques.
5. Les hashes SHA-256 des fichiers source sont enregistres dans `.etl_hashes.json`.
6. Aux lancements suivants, si les fichiers source n'ont pas change, l'ETL est saute.

### 2.3 Separation des Responsabilites

| Couche          | Repertoire         | Responsabilite                                    |
|-----------------|--------------------|---------------------------------------------------|
| Orchestration   | `app.py`           | Navigation, init DB, injection `query_db()`       |
| ETL             | `src/etl/`         | Extraction, transformation, chargement            |
| Agent           | `src/agent/`       | LLM, memoire, RAG, alertes                        |
| Interface       | `src/ui/`          | Composants Streamlit, visualisations Plotly        |
| Machine Learning| `src/ml/`          | Tracking MLflow (wrapper avec fallback gracieux)  |
| Monitoring      | `monitoring/`      | Prometheus, Grafana, alertes regles               |

---

## 3. Choix Technologiques Justifies

### 3.1 PostgreSQL 16 (au lieu de SQLite ou DuckDB)

**Choix** : PostgreSQL 16 dans un conteneur Docker.

**Justification** :
- Support natif de l'extension `pgvector` pour les recherches semantiques (RAG).
- Separation propre entre l'application et le stockage (pas de fichier .db embarque).
- Vues SQL analytiques complexes (jointures multi-tables, fonctions fenetre) plus expressives qu'en Pandas.
- La memoire de l'agent (table `conversations`) et les embeddings (table `review_embeddings`) cohabitent
  dans la meme base, simplifiant la gestion des transactions.
- Reproductible via `docker-compose up -d db` : meme environnement en dev, CI et prod.

**Concretement dans le code** : `src/db_config.py` expose `get_engine()` (SQLAlchemy pool) et
`test_connection()`. Toutes les couches passent par cet unique point d'acces.

### 3.2 Streamlit (au lieu de Flask/FastAPI + React)

**Choix** : Streamlit pour l'interface utilisateur complete.

**Justification** :
- Prototypage rapide de 10 ecrans analytiques sans separer backend et frontend.
- `@st.cache_resource` permet de mettre en cache la connexion DB et l'ETL initial.
- `@st.cache_data` cache les resultats des requetes SQL lourdes.
- Rendu natif de DataFrames, graphiques Plotly, sliders, selectbox sans HTML/CSS.
- Deploiement en une seule commande (`streamlit run app.py`).

**Limite acceptee** : pas adapte a de fortes concurrences (1 thread par session). Acceptable
pour un POC/usage interne.

### 3.3 Plotly (au lieu de Matplotlib ou Seaborn)

**Choix** : Plotly Express et Plotly Graph Objects.

**Justification** :
- Graphiques interactifs (zoom, tooltip, legende cliquable) sans JavaScript supplementaire.
- `px.choropleth` pour la carte geographique des ventes par pays.
- `go.Scatterpolar` pour le radar de comparaison produits.
- Integration native avec `st.plotly_chart()`.

### 3.4 scikit-learn (TF-IDF, Churn, Scoring)

**Choix** : scikit-learn pour trois usages distincts.

**Justification et detail par usage** :

| Usage               | Algorithme                        | Fichier                       |
|---------------------|-----------------------------------|-------------------------------|
| MDM Matching        | TF-IDF + cosine similarity        | `src/etl/mdm_mapping.py`      |
| Churn Clients       | RandomForestClassifier            | `src/ui/analytics.py`         |
| Scoring Produits    | Formule ponderee (pas de fit)     | `src/ui/analytics.py`         |
| Fallback Previsions | LinearRegression                  | `src/ui/analytics.py`         |

Pour le **MDM matching**, TF-IDF sur des n-grammes de caracteres (`analyzer="char_wb"`,
`ngram_range=(2,4)`) est plus robuste aux fautes d'orthographe et aux abreviations que
le matching exact ou Levenshtein. Le seuil de similarite cosinus est fixe a 0.35 :
en dessous, le produit tombe dans le fallback rank-based.

### 3.5 sentence-transformers / pgvector (RAG)

**Choix** : `all-MiniLM-L6-v2` (384 dimensions) + extension pgvector.

**Justification** :
- Vectorisation semantique des avis clients en 384 dimensions.
- Stockage et recherche ANN (Approximate Nearest Neighbor) directement dans PostgreSQL
  via `ivfflat` index  pas de serveur vectoriel supplementaire (Pinecone, Weaviate).
- `ReviewRAG.search("probleme emballage", k=5)` retourne les avis les plus proches
  semantiquement pour contextualiser les reponses de l'agent.

**Condition** : `CREATE EXTENSION IF NOT EXISTS vector` (automatique au premier build_index()).

### 3.6 Multi-LLM avec fallback hors-ligne

**Choix** : Detection automatique du provider via les variables d'environnement, avec
fallback base sur des regles SQL si aucune cle n'est disponible.

**Justification** :
- Groq (llama-3.3-70B) est gratuit et ultra-rapide : provider recommande pour le dev.
- Le fallback SQL garantit que l'application reste fonctionnelle meme sans acces reseau
  ou sans cle API, ce qui est critique pour une demo ou un environnement restreint.
- L'ordre de priorite dans `graph.py` : Groq > Mistral > OpenAI > Anthropic > Fallback.

### 3.7 MLflow (Tracking ML)

**Choix** : MLflow avec wrapper gracieux (`MLTracker`).

**Justification** :
- Tracer les hyperparametres et metriques des modeles churn et forecast (accuracy, RMSE).
- Le wrapper `MLTracker` dans `src/ml/mlflow_tracker.py` absorbe l'`ImportError` si
  mlflow n'est pas installe : l'application tourne normalement sans tracking.
- URI locale par defaut (`mlruns/`) : pas de serveur externe requis.

### 3.8 Prometheus + Grafana (Observabilite)

**Choix** : stack Prometheus/Grafana separee via `docker-compose.monitoring.yml`.

**Justification** :
- Metriques instrumentees dans le code (`monitoring/prometheus_metrics.py`) :
  `etl_runs_total`, `etl_duration_seconds`, `data_quality_checks_total`,
  `agent_requests_total`, `ml_predictions_total`.
- Le serveur de metriques (port 8000) demarre optionnellement au lancement de l'app.
- Grafana pre-provisionne avec un dashboard 6 panneaux et des alertes sur regles.
- Stack optionnelle : l'app principale fonctionne sans elle.

### 3.9 GitHub Actions CI/CD

**Choix** : 4 jobs sequentiels : lint, unit-tests, integration-tests, docker-build.

**Justification** :
- `ruff` (Rust-based) est 10-100x plus rapide que `flake8` ou `pylint`.
- Les integration-tests utilisent un service PostgreSQL ephemere natif GitHub Actions
  (pas de mock) : l'ETL complet est execute et verifie.
- Le build Docker est conditionne a la branche `main` pour eviter les builds inutiles
  sur les branches de feature.

---

## 4. Structure du Projet

```
smartshop360/
|-- app.py                          # Entree Streamlit : 10 ecrans, init DB, routing
|-- requirements.txt                # Dependances Python (commentaires justificatifs)
|-- docker-compose.yml              # Conteneurs PostgreSQL + Streamlit
|-- Dockerfile                      # Image Python 3.11-slim
|-- pyproject.toml                  # Config ruff (linter) + pytest + coverage
|-- pytest.ini                      # Chemins de test, markers
|-- architecture.txt                # Specifications techniques detaillees (source de verite)
|-- .etl_hashes.json                # Hashes SHA-256 des derniers fichiers source ingeres
|-- .env                            # Variables locales (non versionnees)
|
|-- src/                            # Code source principal
|   |-- __init__.py
|   |-- db_config.py                # get_engine(), test_connection()  point d'acces DB unique
|   |
|   |-- etl/
|   |   |-- cleaning.py             # Lecture CSV/JSON, nettoyage, calcul Revenue/Margin
|   |   |-- incremental.py          # SHA-256 des fichiers source, .etl_hashes.json
|   |   |-- data_quality.py         # 7 regles de validation, DataQualityReport
|   |   |-- mdm_mapping.py          # TF-IDF matching + fallback rank-based
|   |   +-- run_etl.py              # Orchestrateur : DDL, ETL complet, vues SQL
|   |
|   |-- agent/
|   |   |-- graph.py                # Detection provider LLM, appels API, SQL generation
|   |   |-- tools.py                # execute_sql(), python_analysis(), SYSTEM_PROMPT
|   |   |-- memory.py               # ConversationMemory  table conversations PostgreSQL
|   |   |-- rag.py                  # ReviewRAG  pgvector + sentence-transformers
|   |   +-- alerts.py               # Alertes email SMTP + Slack webhook
|   |
|   |-- ui/
|   |   |-- dashboard.py            # render_dashboard(), render_product_analysis(), render_data_quality()
|   |   |-- chat.py                 # render_chat()  interface chatbot
|   |   +-- analytics.py           # 6 modules : temporel, geo, comparaison, scoring, churn, forecast
|   |
|   +-- ml/
|       +-- mlflow_tracker.py       # MLTracker  wrapper MLflow avec fallback gracieux
|
|-- monitoring/
|   |-- prometheus_metrics.py       # AppMetrics, start_metrics_server(), decorateur @timed
|   |-- prometheus.yml              # Configuration scrape Prometheus
|   |-- alert_rules.yml             # 4 regles : ETLFailed, ETLSlow, DataQualityErrors, AgentErrorRate
|   |-- docker-compose.monitoring.yml  # Prometheus + Grafana + postgres_exporter
|   +-- grafana/
|       +-- provisioning/
|           |-- dashboards/
|           |   |-- smartshop360.json   # Dashboard 6 panneaux (pre-provisionne)
|           |   +-- dashboard.yml
|           +-- datasources/
|               +-- datasource.yml     # Prometheus comme datasource Grafana
|
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                 # Fixtures : sample_df, db_available, sys.path
|   |-- test_cleaning.py            # 4 classes, 14 methodes : nettoyage CSV
|   |-- test_incremental.py         # 4 classes, 12 methodes : SHA-256, persistance
|   |-- test_data_quality.py        # 8 classes, 20+ methodes : chaque regle
|   +-- test_mdm_mapping.py         # 4 classes, 15 methodes : TF-IDF, mapping
|
|-- .github/
|   +-- workflows/
|       +-- ci.yml                  # 4 jobs : lint, unit-tests, integration, docker-build
|
+-- Data/
    |-- online_retail_II.csv        # Source 1 : 1 067 371 transactions ERP
    +-- labeledReview.datasetFix.json  # Source 2 : 1 000 avis labelises
```

---

## 5. Demarrage Rapide

### Prerequis

- Python 3.11+
- Docker Desktop

### Etape 1 - PostgreSQL via Docker

```bash
docker-compose up -d db
```

Le conteneur PostgreSQL 16 demarre avec les parametres suivants :

| Parametre    | Valeur par defaut |
|--------------|-------------------|
| Hote         | localhost         |
| Port         | 5432              |
| Base         | smartshop_db      |
| Utilisateur  | admin             |
| Mot de passe | password          |

Surcharge possible via `.env` a la racine :

```env
POSTGRES_USER=admin
POSTGRES_PASSWORD=password
POSTGRES_DB=smartshop_db

# LLM (au moins un pour activer l'agent IA)
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Alertes (optionnel)
ALERT_EMAIL_FROM=expediteur@example.com
ALERT_EMAIL_TO=destinataire@example.com
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USER=...
ALERT_SMTP_PASS=...
ALERT_SLACK_WEBHOOK=https://hooks.slack.com/...

# ETL (optionnel)
SAMPLE_SIZE=30000
SEED=42
```

### Etape 2 - Dependances Python

```bash
pip install -r requirements.txt
```

Les packages optionnels (prophet, prometheus-client) sont commentes dans le fichier.
Ils peuvent etre installes separement selon le besoin :

```bash
pip install prophet              # Previsions avancees (long a installer sur Windows)
pip install prometheus-client    # Metriques Prometheus
```

### Etape 3 - Lancer l'application

```bash
streamlit run app.py
```

Au premier lancement, l'ETL s'execute automatiquement (environ 30-60 secondes).
Les lancements suivants sont instantanes si les fichiers source n'ont pas change.

### Etape 4 - (Optionnel) Stack de monitoring

```bash
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

- Prometheus : http://localhost:9090
- Grafana    : http://localhost:3000 (admin / admin)

---

## 6. Pipeline ETL - Detail Complet

### 6.1 Vue Sequentielle

```
online_retail_II.csv          labeledReview.datasetFix.json
        |                                  |
        v                                  v
 load_transactions()                 load_reviews()
 (30 000 lignes, SEED=42)           (1 000 avis, sample)
        |                                  |
        v                                  v
 clean_transactions()            Normalisation sentiment
  - Supprime factures 'C'         (positif/negatif)
  - Filtre Qty > 0, Price > 0           |
  - Supprime CustomerID null            |
  - Calcule Revenue = Qty * Price       |
  - Calcule Margin (taux 20-45%)        |
  - Filtre Revenue > 0                  |
        |                                  |
        v                                  |
 extract_top50_products()                 |
 (Golden Records par volume)             |
        |                                  |
        v                                  |
 build_product_mapping()  <---------------+
 (TF-IDF ERP <-> Avis, seuil 0.35)
 + fallback rank-based si score < seuil
        |
        v
 run_all_validations()
 (10+ regles, severity error/warning)
        |
        v
 Chargement PostgreSQL (SQLAlchemy bulk insert)
  - products (48 Golden Records)
  - customers (4 042 clients)
  - product_mapping (48 mappings)
  - sales_facts (1 670 lignes)
  - review_facts (1 000 avis)
        |
        v
 Creation/mise a jour des 4 vues SQL
  - v_product_kpi
  - v_customer_kpi
  - v_alerts
  - v_data_quality
        |
        v
 record_hashes() -> .etl_hashes.json
```

### 6.2 Detection Incrementale (incremental.py)

Le module `src/etl/incremental.py` calcule le SHA-256 des deux fichiers source
et le compare aux hashes stockes dans `.etl_hashes.json`.

**Comportement** :
- Si les hashes sont identiques et `force=False` : l'ETL est saute entierement.
- Si un fichier a change ou si `force=True` : l'ETL tourne completement.
- A la fin de chaque ETL reussi, les nouveaux hashes sont persistes avec un timestamp.

```python
# Exemple d'utilisation
from src.etl.incremental import should_run_etl, record_hashes

should_run, changed = should_run_etl([CSV_PATH, JSON_PATH], force=False)
# should_run = False si rien n'a change
# should_run = True  si au moins un fichier est different
```

### 6.3 Nettoyage (cleaning.py)

| Operation                           | Colonne(s) concernee(s)        | Raison                                          |
|-------------------------------------|--------------------------------|-------------------------------------------------|
| Suppression factures annulees       | `InvoiceNo` commencant par 'C' | Les retours ne sont pas des ventes              |
| Filtre Quantity > 0                 | `Quantity`                     | Quantites negatives = corrections comptables    |
| Filtre UnitPrice > 0                | `UnitPrice`                    | Prix negatifs = remises hors perimetre          |
| Suppression CustomerID null         | `CustomerID`                   | Pas d'analyse client possible sans ID           |
| Conversion InvoiceDate              | `InvoiceDate`                  | Parsing robuste (format mixte UK/US)            |
| Calcul Revenue = Qty * Price        | Nouveau : `Revenue`            | Metrique principale d'analyse                   |
| Filtre Revenue > 0                  | `Revenue`                      | Double securite apres calcul                    |
| Calcul Margin (taux aleatoire 20-45%)| Nouveau : `Margin`            | Simulation marge (non disponible dans le CSV)   |

La marge est simulee avec `np.random.default_rng(SEED=42)` pour assurer la
reproductibilite. Le SEED est configurable via la variable d'environnement `SEED`.

### 6.4 Golden Records (Top Produits)

L'ETL extrait les 50 produits les plus vendus par volume (`Quantity`) et les enregistre
dans la table `products`. Une categorisation automatique est appliquee par mots-cles :

| Mot-cle dans le nom                    | Categorie assignee           |
|----------------------------------------|------------------------------|
| CANDLE, LIGHT, LAMP, LANTERN, LED      | Luminaires                   |
| BAG, TOTE, PURSE, SATCHEL             | Sacs & Accessoires           |
| MUG, CUP, TEA, COFFEE, PLATE, BOWL   | Art de la Table              |
| FRAME, SIGN, CLOCK, MIRROR            | Decoration Murale            |
| CARD, WRAP, RIBBON, GIFT, BOX         | Emballages & Cadeaux         |
| HEART, ROSE, FLOWER, BIRD, BUTTERFLY  | Nature & Romantique          |
| Autres                                 | Divers                       |

---

## 7. Schema de la Base de Donnees

### 7.1 Tables

#### `products` - Referentiel produit (Golden Records ERP)
```sql
CREATE TABLE products (
    "ProductID"    VARCHAR(50) PRIMARY KEY,   -- StockCode ERP
    "ProductName"  VARCHAR(255),               -- Nom nettoyage
    "Category"     VARCHAR(100)                -- Categorie inferee par mots-cles
);
```

#### `customers` - Clients uniques
```sql
CREATE TABLE customers (
    "ClientID"  VARCHAR(50) PRIMARY KEY,   -- CustomerID du CSV
    "Nom"       VARCHAR(100),               -- Nom genere (anonymise)
    "Pays"      VARCHAR(100)                -- Country du CSV
);
```

#### `sales_facts` - Faits transactionnels
```sql
CREATE TABLE sales_facts (
    "FactID"       SERIAL PRIMARY KEY,
    "InvoiceNo"    VARCHAR(20),       -- Numero de facture
    "StockCode"    VARCHAR(50),       -- Cle etrangere vers products
    "Quantity"     INTEGER,
    "Revenue"      NUMERIC(12,2),     -- Qty * UnitPrice
    "Margin"       NUMERIC(12,2),     -- Revenue * taux_marge
    "InvoiceDate"  TIMESTAMP,
    "CustomerID"   VARCHAR(50)        -- Cle etrangere vers customers
);
```

#### `review_facts` - Avis clients
```sql
CREATE TABLE review_facts (
    "ReviewID"    SERIAL PRIMARY KEY,
    "ProductID"   VARCHAR(50),        -- Code produit issu du dataset avis
    "Rating"      NUMERIC(3,1),       -- Note de 1 a 5
    "ReviewText"  TEXT,               -- Texte brut de l'avis
    "Sentiment"   VARCHAR(20),        -- positive / negative / neutral
    "ReviewDate"  TIMESTAMP
);
```

#### `product_mapping` - Table pivot MDM
```sql
CREATE TABLE product_mapping (
    "MappingID"           SERIAL PRIMARY KEY,
    "ERP_StockCode"       VARCHAR(50),    -- StockCode de l'ERP
    "ERP_ProductName"     VARCHAR(255),   -- Nom ERP
    "Review_ProductCode"  VARCHAR(50),    -- Code produit source avis
    "Review_ProductName"  VARCHAR(255),   -- Nom produit source avis
    "Category"            VARCHAR(100),
    "GoldenRecordName"    VARCHAR(255),   -- Nom unifie (Golden Record)
    "MatchScore"          FLOAT,          -- Score cosinus TF-IDF (0 a 1)
    "MatchStrategy"       VARCHAR(20)     -- 'tfidf' ou 'rank'
);
```

#### `conversations` - Memoire persistante de l'agent
```sql
CREATE TABLE conversations (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(100) NOT NULL,  -- UUID de session Streamlit
    role        VARCHAR(20)  NOT NULL,  -- 'user' ou 'assistant'
    content     TEXT         NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_conv_session ON conversations(session_id, created_at DESC);
```

### 7.2 Vues Analytiques

#### `v_product_kpi` - KPIs produits croises
Jointure entre `product_mapping`, `sales_facts` et `review_facts`.
Colonnes : ProductID, ProductName, Category, CA, Marge, QuantiteVendue,
NoteMoyenne, NbAvis, AvisPositifs, AvisNegatifs, AvisNeutres.

#### `v_customer_kpi` - KPIs clients
Jointure entre `customers` et `sales_facts`.
Colonnes : ClientID, Nom, Pays, NbCommandes, CA_Total, PanierMoyen.

#### `v_alerts` - Produits a surveiller
Calcul du statut a partir de `v_product_kpi` :
- `CRITIQUE`     : NoteMoyenne < 3.0 ET QuantiteVendue > 50
- `A_SURVEILLER` : NoteMoyenne < 3.5
- `OK`           : sinon

#### `v_data_quality` - Metriques de couverture MDM
Nb_Produits_ERP, Nb_Produits_Avis, Nb_Mappings, Nb_Avis_Total,
Nb_Avis_Lies, Nb_Factures, Nb_Clients, Taux_Couverture_MDM (%).

---

## 8. Strategie MDM et Matching Produit

### 8.1 Le Probleme

Les deux sources de donnees n'ont aucun identifiant commun :
- L'ERP identifie les produits par `StockCode` (ex: "85123A").
- Le dataset d'avis identifie les produits par un code different et un nom libre.

La reconciliation manuelle est impossible a l'echelle. Une strategie automatique est
necessaire pour creer la table `product_mapping` (Golden Records).

### 8.2 Niveaux de Matching (par priorite)

```
Niveau 1 : Hard match (cle naturelle commune EAN/GTIN)
             -> Deterministe, aucune ambiguite
             -> Non applicable sur ces datasets (pas de code commun)

Niveau 2 : Fuzzy match TF-IDF + cosine similarity
             -> TF-IDF sur n-grammes de caracteres (char_wb, 2-4)
             -> Score cosinus calcule pour chaque paire ERP x Avis
             -> Seuil : score >= 0.35 -> match accepte
             -> Colonne MatchStrategy = 'tfidf', MatchScore = score cosinus

Niveau 3 : Embedding semantique (sentence-transformers)
             -> Disponible via src/agent/rag.py (ReviewRAG)
             -> Non applique dans l'ETL principal (cout calcul)

Niveau 4 : Rank-based fallback
             -> Produit ERP i -> Produit Avis i (par rang de volume)
             -> Colonne MatchStrategy = 'rank', MatchScore = 0.0
             -> Utilise quand score TF-IDF < 0.35
```

### 8.3 Detail du TF-IDF (mdm_mapping.py)

```python
vectorizer = TfidfVectorizer(
    analyzer    = "char_wb",   # n-grammes de caracteres (inclut espaces)
    ngram_range = (2, 4),      # bigrammes a 4-grammes de caracteres
    lowercase   = True,        # normalisation casse
)
```

Le choix de `char_wb` plutot que `word` est delibere : il rend le matching robuste
aux fautes d'orthographe, aux abreviations et aux variations de formatage
(ex: "CERAMIC MUG" <-> "Ceramic-mug" -> score eleve meme sans mot commun exact).

La matrice de similarite est de forme `(n_erp, n_review)`. Pour chaque produit ERP,
le produit avis avec le score maximum est retenu si ce score depasse le seuil.

### 8.4 Resultats sur le Dataset Actuel

| Metrique                       | Valeur |
|--------------------------------|--------|
| Produits ERP (Golden Records)  | 48     |
| Mappings TF-IDF (score >= 0.35)| variable selon run |
| Mappings rank-based (fallback) | complement        |
| Total mappings                 | 48     |
| Taux couverture MDM            | calcule dans v_data_quality |

---

## 9. Qualite des Donnees

### 9.1 Architecture de Validation (data_quality.py)

Le module `src/etl/data_quality.py` implemente un moteur de validation inspire de
Great Expectations, sans aucune dependance externe. Chaque regle retourne un
`ExpectationResult` avec un niveau de severite (`error` ou `warning`).

```
DataQualityReport
  |-- ExpectationResult (rule, passed, detail, severity)
  |-- ExpectationResult ...
  |
  |-- .passed  -> True si aucune erreur de severite 'error'
  |-- .errors  -> liste des echecs bloquants
  +-- .warnings -> liste des echecs non-bloquants
```

### 9.2 Regles Implementees

| Fonction                     | Description                              | Severite par defaut |
|------------------------------|------------------------------------------|---------------------|
| `expect_no_nulls(df, col)`   | Aucune valeur nulle dans la colonne      | error               |
| `expect_min_rows(df, n)`     | Au moins n lignes dans le DataFrame      | error               |
| `expect_unique(df, col)`     | Aucun doublon dans la colonne            | warning             |
| `expect_values_in_set(df, col, vals)` | Toutes les valeurs dans l'ensemble | error            |
| `expect_column_between(df, col, min, max)` | Valeurs dans la plage [min, max]| error          |
| `expect_no_future_dates(df, col)` | Aucune date future                  | warning             |
| `expect_positive_values(df, col)` | Toutes les valeurs > 0              | warning             |

### 9.3 Comportement dans l'ETL

Le comportement en cas d'echec depend du parametre `force` :

```python
if not quality_ok:
    if force:
        print("Erreurs qualite detectees  force=True, ingestion poursuivie.")
    else:
        raise ValueError("Validation qualite des donnees echouee.")
```

- `run_etl(force=False)` : bloquant en cas d'erreur de severite `error`.
- `run_etl(force=True)` : non-bloquant (utilise au premier lancement depuis `app.py`).

La regle `expect_positive_values(df, "Revenue")` est configuree en `severity="warning"`
car quelques lignes a Revenue nul peuvent subsister apres nettoyage sans invalider
l'ensemble du dataset.

---

## 10. Agent IA Text-to-SQL

### 10.1 Architecture de l'Agent (graph.py)

L'agent transforme une question en langage naturel en requete SQL PostgreSQL,
execute la requete, et formate la reponse.

```
Question utilisateur
        |
        v
_detect_provider()
  -> determine le provider LLM (Groq/Mistral/OpenAI/Anthropic/Fallback)
        |
        v
SYSTEM_PROMPT (src/agent/tools.py)
  -> schema BDD complet (tables + vues)
  -> instructions de formatage SQL
  -> exemples de requetes
        |
        v
Appel API LLM (HTTP requests, pas de SDK)
  -> Groq   : https://api.groq.com/openai/v1/chat/completions
  -> Mistral: https://api.mistral.ai/v1/chat/completions
  -> etc.
        |
        v
Extraction JSON+SQL (regex sur bloc JSON contenant la cle "sql")
        |
        v
execute_sql() -> DataFrame Pandas
        |
        v
Reponse formate + SQL affiche dans l'UI
```

### 10.2 Providers LLM Supportes

| Provider   | Modele                      | Detection                              | Variable          |
|------------|-----------------------------|----------------------------------------|-------------------|
| Groq       | llama-3.3-70b-versatile     | cle commence par `gsk_`                | GROQ_API_KEY      |
| Anthropic  | claude-3-5-haiku-20241022   | cle commence par `sk-ant-`             | ANTHROPIC_API_KEY |
| OpenAI     | gpt-4o-mini                 | cle commence par `sk-`                 | OPENAI_API_KEY    |
| Mistral    | mistral-large-latest        | cle 28-64 chars, ne commence pas par sk| MISTRAL_API_KEY   |
| Fallback   | Regles SQL par mots-cles    | aucune cle disponible                  | -                 |

La detection est automatique : si une cle est passee en parametre, elle prend la priorite
sur les variables d'environnement. L'ordre de fallback suit la liste ci-dessus.

### 10.3 Mode Fallback Hors-Ligne

Sans cle API, l'agent detecte des mots-cles dans la question et execute des requetes SQL
predefinies sur les vues analytiques (v_product_kpi, v_customer_kpi, v_alerts).
Ce mode couvre les questions les plus frequentes sans aucun appel reseau.

---

## 11. Memoire et RAG

### 11.1 Memoire Persistante (memory.py)

La classe `ConversationMemory` stocke l'historique de chaque session dans la table
`conversations` de PostgreSQL. Cela permet a l'agent de maintenir le contexte entre
les questions d'une meme session Streamlit.

```python
mem = ConversationMemory(session_id="uuid_session")
mem.add("user", "Quels sont nos best-sellers ?")
mem.add("assistant", "Les 5 meilleurs produits par CA sont...")
history = mem.get_history(last_n=10)
# retourne [{"role": "user", "content": "..."}, ...]
```

L'index `idx_conv_session` sur `(session_id, created_at DESC)` optimise la recuperation
des N derniers messages.

### 11.2 RAG sur les Avis (rag.py)

La classe `ReviewRAG` permet de rechercher les avis clients les plus pertinents
semantiquement pour enrichir les reponses de l'agent.

**Construction de l'index** (a faire une fois apres l'ETL) :
```bash
python -c "from src.agent.rag import ReviewRAG; ReviewRAG().build_index()"
```

**Architecture** :
- Les textes des avis (`review_facts.ReviewText`) sont vectorises par `all-MiniLM-L6-v2` (384 dimensions).
- Les vecteurs sont stockes dans `review_embeddings` (colonne `vector(384)`).
- La recherche utilise l'index `ivfflat` de pgvector (ANN par cosine distance).
- `rag.search("probleme livraison", k=5)` retourne les 5 avis les plus proches.

---

## 12. Fonctionnalites par Ecran

### 12.1 Ecrans Principaux (src/ui/dashboard.py)

#### Dashboard (Ecran 1)
- KPIs globaux en cartes metriques : CA total, nombre de clients, nombre de produits, note moyenne.
- Tableau filtrable des produits avec colonne Statut (CRITIQUE / A_SURVEILLER / OK).
- Section alertes : produits en statut CRITIQUE issus de `v_alerts`.
- Boutons d'export CSV/Excel du tableau filtre.

#### Analyse Produit 360 (Ecran 2)
- Selecteur de produit parmi les Golden Records.
- Fiche complete : CA, Marge, Quantite vendue, Note moyenne, NbAvis.
- Courbe d'evolution du CA dans le temps (par mois).
- Verbatims des derniers avis (texte brut + sentiment + note).
- Repartition des sentiments (pie chart).

#### Qualite des Donnees (Ecran 4)
- Metriques de couverture MDM depuis `v_data_quality`.
- Tableau de la table `product_mapping` avec MatchScore et MatchStrategy.
- Documentation de la strategie MDM (niveaux de matching).

### 12.2 Modules Analytics Avances (src/ui/analytics.py)

#### Analyse Temporelle (Ecran 5)
- Filtres date debut / date fin avec validation.
- Selecteur de granularite : Mensuelle / Trimestrielle / Annuelle.
- Courbes CA, Quantite vendue, Nombre de factures (triple axe).
- Requete SQL directe sur `sales_facts` avec `DATE_TRUNC`.

#### Carte Geographique (Ecran 6)
- Choropleth mondial des ventes par pays (Plotly `px.choropleth`).
- Selecteur de metrique : CA total, Nombre de commandes, Nombre de clients uniques.
- Tableau de classement des pays.
- Requete SQL sur `sales_facts JOIN customers`.

#### Comparaison Produits (Ecran 7)
- Selection de 2 a 4 produits (multiselect).
- Graphique radar (Plotly `go.Scatterpolar`) sur 5 axes : CA, Note, Quantite, AvisPos, AvisNeg.
- Graphique barres groupees pour comparaison directe des valeurs absolues.
- Tableau de comparaison (fix Arrow : `.astype(str)` avant `st.dataframe()`).

#### Scoring Produits (Ecran 8)
Formule de score composite :

  Score = 0.40 * NormCA + 0.30 * NormNote + 0.20 * NormQte + 0.10 * (1 - NormAvisNeg)

Chaque composante est normalisee entre 0 et 1 (min-max scaling). Le score final est dans [0, 1].
Affichage : bar chart Top 10 + scatter CA vs Note (taille = quantite) + tableau trie.

#### Churn Clients (Ecran 9)
Pipeline RFM + classification :
1. Calcul de `recence_jours` = date_reference - derniere_commande.
   La date de reference est `max(derniere_commande)` (relative aux donnees, pas `now()`).
2. Slider dynamique pour le seuil d'inactivite (defaut = percentile 70% de recence_jours).
3. Label binaire : `churn = 1` si recence_jours > seuil.
4. Features : recence_jours, frequence (nb commandes), montant (CA total), duree_relation (jours).
5. RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced").
6. Affichage : importance des features, distribution des probabilites de churn, top 20 clients a risque.
7. Si classe unique : histogramme RFM + message explicatif pour ajuster le seuil.

#### Prevision Ventes (Ecran 10)
1. Tentative d'utilisation de Prophet (`from prophet import Prophet`).
2. Si Prophet absent : regression lineaire sklearn (LinearRegression sur feature `t`).
   La prediction utilise `pd.DataFrame({"t": ...})` pour eviter l'avertissement
   `X does not have valid feature names`.
3. Affichage : courbe historique + prevision + intervalle de confiance.
4. Tracking MLflow si disponible (MLTracker).

---

## 13. Machine Learning

### 13.1 Modele de Churn (RandomForestClassifier)

| Parametre         | Valeur                   |
|-------------------|--------------------------|
| Algorithm         | RandomForestClassifier   |
| n_estimators      | 100                      |
| random_state      | 42                       |
| Features          | recence_jours, frequence, montant, duree_relation |
| Label             | churn (0/1, seuil dynamique = percentile 70%) |
| Split             | 70% train / 30% test     |

### 13.2 Scoring Produit (Formule Ponderee)

Aucun entrainement requis. La formule est deterministe et transparente :

| Composante         | Poids | Source                  |
|--------------------|-------|-------------------------|
| CA normalise       | 0.40  | v_product_kpi."CA"      |
| Note moyenne norm. | 0.30  | v_product_kpi."Notemoyenne" |
| Quantite norm.     | 0.20  | v_product_kpi."QuantiteVendue" |
| (1 - AvisNeg norm.)| 0.10  | v_product_kpi."AvisNegatifs" |

### 13.3 Tracking MLflow (mlflow_tracker.py)

Le wrapper `MLTracker` encapsule MLflow avec un fallback gracieux :

```python
with MLTracker("churn_model") as tracker:
    tracker.log_params({"n_estimators": 100, "threshold": seuil})
    tracker.log_metrics({"accuracy": acc, "f1": f1_score})
    tracker.log_model(clf, "churn_rf")
```

Si MLflow n'est pas installe, les appels sont silencieusement ignores.
L'URI de tracking par defaut est `mlruns/` (local). Configurable via `MLFLOW_TRACKING_URI`.

---

## 14. Observabilite et Alertes

### 14.1 Metriques Prometheus (prometheus_metrics.py)

```
Metrique                         Type       Description
--------------------------------------------------------------
etl_runs_total                   Counter    Nombre de runs ETL
etl_duration_seconds             Histogram  Duree des runs ETL
data_quality_checks_total        Counter    Checks passes/echoues
sql_queries_total                Counter    Requetes SQL executees
agent_requests_total             Counter    Appels a l'agent IA
alerts_triggered_total           Counter    Alertes declenchees
ml_predictions_total             Counter    Predictions ML
```

Le serveur de metriques demarre sur le port 8000 au lancement de `app.py` si
`prometheus-client` est installe. Sinon, l'exception est silencieusement ignoree.

### 14.2 Regles d'Alerte (alert_rules.yml)

| Alerte               | Condition                                  | Severite |
|----------------------|--------------------------------------------|----------|
| ETLFailed            | etl_runs_total{status="failure"} > 0       | critical |
| ETLSlow              | etl_duration_seconds > 300                 | warning  |
| DataQualityErrors    | data_quality_checks_total{status="failed"} > 5 | warning |
| AgentHighErrorRate   | agent_requests_total{status="error"} > 10  | warning  |

### 14.3 Alertes Email et Slack (alerts.py)

Le module `src/agent/alerts.py` interoge `v_alerts` et envoie des notifications
pour les produits en statut CRITIQUE :

- **Email** : SMTP via `smtplib` (configurable via variables d'environnement).
- **Slack** : HTTP POST sur le webhook configuré via `ALERT_SLACK_WEBHOOK`.

---

## 15. Tests et CI/CD

### 15.1 Lancer les Tests

```bash
# Tests unitaires (ne necessitent pas PostgreSQL)
pytest tests/ --ignore=tests/test_integration.py -v

# Avec couverture
pytest tests/ --ignore=tests/test_integration.py --cov=src --cov-report=term-missing
```

### 15.2 Detail des Suites de Tests

| Fichier                  | Classes | Methodes | Perimetre                                      |
|--------------------------|---------|----------|------------------------------------------------|
| `test_cleaning.py`       | 4       | 14       | load_transactions, clean_transactions, extract_top50, extract_customers |
| `test_incremental.py`    | 4       | 12       | _file_sha256, should_run_etl, record_hashes    |
| `test_data_quality.py`   | 8       | 20+      | Chaque regle individuellement (passed/failed)  |
| `test_mdm_mapping.py`    | 4       | 15       | fuzzy_match_tfidf, build_product_mapping       |

Les fixtures communes sont dans `tests/conftest.py` :
- `sample_df` : DataFrame de 100 lignes representatif pour les tests de nettoyage.
- `db_available` : booleen pour skipper les tests neccessitant PostgreSQL.

### 15.3 Pipeline CI/CD (ci.yml)

```
Push / Pull Request
      |
      v
Job 1: lint
  ruff check src/ tests/
  (echec bloquant)
      |
      v
Job 2: unit-tests
  pytest tests/ --ignore=tests/test_integration.py
  coverage >= 60% requis
  rapport codecov
      |
      v
Job 3: integration-tests
  Service PostgreSQL 15 ephemere (GitHub Actions)
  python -m src.etl.run_etl
  verification tables non vides
      |
      v (branche main uniquement)
Job 4: docker-build
  docker build -t smartshop360:latest .
  (validation image)
```

---

## 16. Deploiement Docker

### 16.1 Application seule

```bash
docker-compose up -d
```

Demarre deux services :
- `db` : PostgreSQL 16, port 5432, volume `pgdata` persistant.
- `app` : Streamlit, port 8501, monte le repertoire courant en volume.

### 16.2 Avec Stack de Monitoring

```bash
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

Services supplementaires :
- `prometheus` : port 9090, scrape l'app sur le port 8000.
- `grafana` : port 3000 (admin/admin), dashboard pre-provisionne.
- `postgres_exporter` : expose les metriques internes PostgreSQL a Prometheus.

---

## 17. KPIs du Projet

| Metrique                           | Valeur        |
|------------------------------------|---------------|
| Lignes brutes CSV                  | 1 067 371     |
| Echantillon utilise (SEED=42)      | 30 000        |
| Lignes apres nettoyage             | ~22 500       |
| Golden Records (Top produits)      | 48            |
| Clients uniques                    | 4 042         |
| Mappings MDM                       | 48            |
| Avis clients                       | 1 000         |
| Lignes de ventes chargees          | 1 670         |
| Vues SQL analytiques               | 4             |
| Providers LLM supportes            | 5             |
| Ecrans Streamlit                   | 10            |
| Regles de qualite ETL              | 10+           |
| Suites de tests unitaires          | 4 (60+ methodes)|
| Jobs CI/CD                         | 4             |

---

## 18. Exemples de Questions pour l'Agent

L'agent utilise les vues `v_product_kpi`, `v_customer_kpi` et `v_alerts` pour
repondre aux questions metier :

```sql
-- Questions produit
"Quels produits vendus a plus de 50 unites ont une note inferieure a 3 ?"
"Quels sont nos 5 best-sellers en chiffre d'affaires ?"
"Quels produits ont beaucoup de ventes mais des avis majoritairement negatifs ?"
"Quelle est la note moyenne par categorie ?"

-- Questions client
"Quels segments de clients sont les plus rentables ?"
"Quels clients n'ont pas commande depuis plus de 6 mois ?"
"Quel est le panier moyen par pays ?"

-- Questions qualite et MDM
"Quel est le taux de couverture MDM de notre referentiel produit ?"
"Combien de produits sont en statut CRITIQUE ?"

-- Questions temporelles
"Quelle est la tendance du CA sur les 6 derniers mois ?"
"Quels mois ont les meilleures ventes ?"
```

---

## Technologies Utilisees

| Composant          | Technologie                           | Justification                                   |
|--------------------|---------------------------------------|-------------------------------------------------|
| Interface          | Streamlit >= 1.32                     | Prototypage rapide, cache_resource/data natif   |
| Visualisation      | Plotly >= 5.18                        | Interactif, choropleth, radar, sans JS custom   |
| Base de donnees    | PostgreSQL 16 + pgvector              | Vues SQL, transactions, index vectoriels        |
| ORM                | SQLAlchemy >= 2.0                     | Pool de connexions, compatibilite multi-DB      |
| Pipeline ETL       | Pandas >= 2.0, NumPy >= 1.24          | Manipulation DataFrames, calculs vectorises     |
| Machine Learning   | scikit-learn >= 1.3                   | TF-IDF, RandomForest, LinearRegression          |
| Previsions         | statsmodels >= 0.14 / Prophet (opt.)  | ARIMA fallback / Prophet si installe            |
| Embeddings RAG     | sentence-transformers (MiniLM-L6-v2)  | Vectorisation avis, 384 dim, leger et rapide    |
| Tracking ML        | MLflow >= 2.10 (optionnel)            | Experiments, params, metriques, modeles         |
| Metriques          | prometheus-client (optionnel)         | Compteurs/histogrammes, scrape Prometheus       |
| Export             | openpyxl >= 3.1                       | Export Excel depuis DataFrames Pandas           |
| Conteneurisation   | Docker + docker-compose               | Reproductibilite env. dev/CI/prod               |
| Tests              | pytest + pytest-cov                   | Unitaires, fixtures, couverture                 |
| Linter             | ruff                                  | 10-100x plus rapide que flake8                  |
| CI/CD              | GitHub Actions                        | lint + tests + integration + docker             |
