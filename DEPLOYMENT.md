# SmartShop 360 — Guide de Déploiement

> **Stack** : Python 3.11 · PostgreSQL 16 · Streamlit · Ollama / Groq · MLflow · Docker

---

## Table des matières

1. [Prérequis](#1-prérequis)
2. [Configuration `.env`](#2-configuration-env)
3. [Mode Développement local](#3-mode-développement-local)
4. [Mode Docker (recommandé)](#4-mode-docker-recommandé)
5. [Initialisation de la base de données](#5-initialisation-de-la-base-de-données)
6. [Pipeline ETL](#6-pipeline-etl)
7. [Agent LLM — Ollama + Groq](#7-agent-llm--ollama--groq)
8. [MLflow](#8-mlflow)
9. [Déploiement production](#9-déploiement-production)
10. [Vérifications et dépannage](#10-vérifications-et-dépannage)

---

## 1. Prérequis

| Outil | Version minimale | Vérification |
|-------|-----------------|--------------|
| Python | 3.11 | `python --version` |
| Docker Desktop | 4.x | `docker --version` |
| Docker Compose | v2 | `docker compose version` |
| Git | 2.x | `git --version` |
| Ollama *(optionnel)* | 0.3+ | `ollama --version` |

**Fichiers de données** requis dans `Data/` :
```
Data/
  online_retail_II.csv          # Transactions (source : Kaggle UCI)
  labeledReview.datasetFix.json # Avis clients labelisés
```

---

## 2. Configuration `.env`

```bash
# Copier et adapter le fichier d'environnement
cp .env .env.local   # garder .env pour référence
```

Variables **obligatoires** à personnaliser :

```dotenv
# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<mot_de_passe_fort>
POSTGRES_DB=smartshop_db
POSTGRES_HOST=localhost          # 'db' si lancé via Docker Compose
POSTGRES_PORT=5432

# LLM — au moins une clé (Ollama local OU Groq API)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
GROQ_API_KEY=gsk_...             # https://console.groq.com

# ETL — volume de données
SAMPLE_SIZE=30000                # 0 = toutes les données (~1M lignes, ~10 min)
TOP_N_PRODUCTS=50                # 0 = tous les produits uniques

# MDM
MDM_STRATEGY=semantic            # poc | fuzzy | semantic
```

> **Sécurité** : `.env` est dans `.gitignore`. Ne jamais le committer avec des vraies clés.

---

## 3. Mode Développement local

### 3.1 Environnement Python

```powershell
# Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### 3.2 Lancer PostgreSQL (Docker uniquement pour la BDD)

```powershell
# Démarre uniquement le conteneur PostgreSQL
docker compose up -d db

# Vérifier que la BDD est ready
docker compose ps
```

### 3.3 Initialiser la base de données

```powershell
python scripts/setup_db.py         # Crée tables + index + vues
```

### 3.4 Lancer l'ETL

```powershell
python -m src.etl.run_etl
# durée : ~30 s (SAMPLE_SIZE=30000) | ~10-15 min (SAMPLE_SIZE=0)
```

### 3.5 Lancer l'application

```powershell
streamlit run app.py
# → http://localhost:8501
```

---

## 4. Mode Docker (recommandé)

Un seul `docker compose up` orchestre PostgreSQL + Streamlit + ETL automatique.

```powershell
# Build + démarrage complet
docker compose up --build

# En arrière-plan
docker compose up --build -d

# Logs en temps réel
docker compose logs -f app

# Arrêt propre (conserve les données)
docker compose down

# Arrêt + suppression des données
docker compose down -v
```

**URL** : http://localhost:8501

### Ordre de démarrage

```
db (PostgreSQL) → [healthcheck OK] → app (ETL + Streamlit)
```

L'application attend que PostgreSQL soit `healthy` (via `pg_isready`) avant de lancer l'ETL.

### Variables d'environnement Docker

Dans `docker-compose.yml`, `POSTGRES_HOST` est automatiquement redéfini à `db` (nom du service Docker) pour que l'app pointe vers le bon conteneur :

```yaml
environment:
  POSTGRES_HOST: db   # override du .env
```

---

## 5. Initialisation de la base de données

Le script `scripts/setup_db.py` gère l'initialisation complète.

```powershell
# Initialisation complète (drop + recreate)
python scripts/setup_db.py

# Recréer uniquement les vues (après modification de VIEWS_SQL)
python scripts/setup_db.py --views-only

# Tout supprimer sans recréer
python scripts/setup_db.py --drop-only

# Vérifier l'état actuel (lecture seule)
python scripts/setup_db.py --check
```

**Objets créés** :

| Type | Nom | Description |
|------|-----|-------------|
| Extension | `pg_trgm` | Recherche floue sur VARCHAR |
| Extension | `unaccent` | Normalisation d'accents |
| Table | `products` | Catalogue ERP |
| Table | `customers` | Référentiel clients (pays en anglais) |
| Table | `product_mapping` | Golden Records MDM (P1→P5) |
| Table | `sales_facts` | Transactions de ventes |
| Table | `review_facts` | Avis clients + sentiment |
| Index | 11 index | dont 2 GIN trigram sur les noms produits |
| Vue | `v_product_kpi` | KPIs produit : CA, marge, notes |
| Vue | `v_customer_kpi` | KPIs client : commandes, panier, note |
| Vue | `v_alerts` | Alertes CRITIQUE / A_SURVEILLER / OK |
| Vue | `v_data_quality` | Tableau de bord qualité MDM |

---

## 6. Pipeline ETL

```powershell
# Lancement normal (détection incrémentale — ignore si données inchangées)
python -m src.etl.run_etl

# Forcer le rechargement complet
python -m src.etl.run_etl --force
```

### Étapes du pipeline

```
1. Nettoyage CSV        cleaning.py       → transactions nettoyées
2. Mapping MDM          mdm_mapping.py    → Golden Records (P1→P5 SBERT)
3. Chargement PG        run_etl.py        → 5 tables chargées
4. Création des vues    run_etl.py        → 4 vues analytiques
5. Validation qualité   data_quality.py   → checks + alertes
6. Tracking MLflow      mlflow_tracker.py → métriques loggées
```

### Stratégies MDM (`MDM_STRATEGY`)

| Valeur | Pipeline actif | Usage |
|--------|---------------|-------|
| `poc` | P5 rank-based uniquement | Démo rapide |
| `fuzzy` | P1 + P2 (rapidfuzz) + P4 (TF-IDF) | Sans GPU |
| `semantic` | P1 + P2 + P3 (SBERT) + P4 + P5 | **Production** |

---

## 7. Agent LLM — Ollama + Groq

### Chaîne de priorité

```
Ollama local  →  Groq (fallback)  →  Mistral / OpenAI / Anthropic  →  Règles SQL
```

### 7.1 Ollama (recommandé — 100% local, gratuit)

```powershell
# Installation : https://ollama.com/download

# Télécharger le modèle (une seule fois, ~2 GB)
ollama pull llama3.2

# Vérifier que le serveur tourne (port 11434)
ollama serve          # ou lancé automatiquement en background

# Tester
curl http://localhost:11434/api/tags
```

Le code détecte automatiquement Ollama via `GET /api/tags` (timeout 1 s).
Si Ollama est absent ou en erreur, bascule automatiquement sur Groq.

**Modèles disponibles** (adapter `OLLAMA_MODEL` dans `.env`) :

| Modèle | RAM requise | Qualité SQL |
|--------|------------|-------------|
| `llama3.2:1b` | 1 GB | ⭐⭐ |
| `llama3.2` | 4 GB | ⭐⭐⭐ |
| `llama3.1:8b` | 8 GB | ⭐⭐⭐⭐ |
| `qwen2.5-coder:7b` | 8 GB | ⭐⭐⭐⭐⭐ (SQL) |

### 7.2 Groq (fallback cloud — gratuit, rapide)

1. Créer un compte sur [console.groq.com](https://console.groq.com)
2. Générer une clé API (préfixe `gsk_`)
3. Ajouter dans `.env` :
   ```dotenv
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

---

## 8. MLflow

### 8.1 Lancer le serveur MLflow

```powershell
# Windows PowerShell
.\start_mlflow.ps1

# Ou manuellement
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

**UI** : http://localhost:5000

### 8.2 Configuration `.env`

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db   # local
# MLFLOW_BACKEND_STORE_URI=postgresql://user:pwd@host/mlflow_db  # prod
MLFLOW_ARTIFACT_ROOT=mlartifacts
```

### 8.3 Expériences trackées

| Expérience | Contenu |
|-----------|---------|
| `SmartShop360/ETL` | Durée ETL, nb lignes, taux de couverture MDM |
| `SmartShop360/ETL > tfidf_matching_*` | Métriques TF-IDF (run imbriqué) |

---

## 9. Déploiement production

### 9.1 Variables `.env` à changer pour la prod

```dotenv
APP_ENV=prod
POSTGRES_HOST=<ip_serveur_postgres>
POSTGRES_PASSWORD=<mot_de_passe_fort>
SAMPLE_SIZE=0                    # toutes les données
TOP_N_PRODUCTS=0

# MLflow avec backend PostgreSQL
MLFLOW_BACKEND_STORE_URI=postgresql://user:pwd@host:5432/mlflow_db
MLFLOW_ARTIFACT_ROOT=s3://mon-bucket/mlflow    # ou volume NFS

MDM_STRATEGY=semantic
```

### 9.2 `docker-compose.yml` production

```yaml
# Remplacer le service 'app' par :
app:
  image: smartshop360:latest     # image buildée en CI/CD
  restart: always
  deploy:
    replicas: 2
  environment:
    POSTGRES_HOST: db
  env_file: .env.prod
```

### 9.3 Checklist avant mise en production

- [ ] PostgreSQL avec mot de passe fort (`POSTGRES_PASSWORD`)
- [ ] Volume Docker persistant (`pgdata`) ou base RDS/Cloud SQL
- [ ] Clé API LLM configurée (Groq ou autre)
- [ ] `SAMPLE_SIZE=0` et `TOP_N_PRODUCTS=0`
- [ ] MLflow avec backend PostgreSQL (pas SQLite)
- [ ] `.env` **non commité** (vérifier `.gitignore`)
- [ ] `python scripts/setup_db.py` exécuté une seule fois avant l'ETL
- [ ] ETL lancé : `python -m src.etl.run_etl`
- [ ] Vues vérifiées : `python scripts/setup_db.py --check`

---

## 10. Vérifications et dépannage

### Commandes de diagnostic

```powershell
# État de la BDD (tables, vues, index)
python scripts/setup_db.py --check

# Test de connexion PostgreSQL
python -c "from src.db_config import test_connection; print(test_connection())"

# Test de l'agent LLM
python -c "
from src.agent.graph import get_active_provider, run_agent
print('Provider:', get_active_provider())
r = run_agent('Quels sont les 3 meilleurs produits par CA ?')
print('SQL:', r['sql'])
print('Réponse:', r['answer'][:200])
"

# Vérifier Ollama
curl http://localhost:11434/api/tags

# Logs Docker
docker compose logs -f app
docker compose logs db
```

### Problèmes fréquents

| Symptôme | Cause probable | Solution |
|----------|---------------|---------|
| `could not connect to server` | PostgreSQL non démarré | `docker compose up -d db` |
| `SSL error` / `certificate` | psycopg2 strict | Ajouter `sslmode=disable` dans `.env` |
| ETL ignoré au redémarrage | Détection incrémentale | `python -m src.etl.run_etl --force` |
| Agent 0 résultats pour un pays | Pays en français dans la requête | Les pays sont en **anglais** dans la BDD : `Australia`, `United Kingdom`… |
| `relation "v_customer_kpi" already exists` | Vue avec colonnes modifiées | `python scripts/setup_db.py --views-only` |
| MLflow `500 search_datasets` | Backend FileStore | Utiliser `MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db` |
| `OOM` avec `MDM_STRATEGY=semantic` | SBERT charge le modèle SBERT | Passer à `MDM_STRATEGY=fuzzy` ou augmenter la RAM |

### Réinitialisation complète

```powershell
# Tout supprimer et repartir de zéro
docker compose down -v                  # supprime les volumes Docker
python scripts/setup_db.py              # recrée tables + vues
python -m src.etl.run_etl --force       # recharge toutes les données
```
